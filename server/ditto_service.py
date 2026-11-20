"""Experimental Ditto avatar service, API-compatible with AI_Mirror's bridge.

Run from the repo root with .venv-ditto on port 8011. The normal mirror server
uses it by setting LIVETALKING_URL=http://127.0.0.1:8011 before starting.
"""
import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import math
import os
from pathlib import Path
import pickle
import queue
import sys
import threading
import time
import types
import wave

import av
import numpy as np
import torch
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from fastapi import FastAPI, Request, UploadFile, File, Form
from fractions import Fraction

ROOT = Path(__file__).resolve().parent.parent
DITTO = ROOT / "vendor" / "ditto-talkinghead"
sys.path.insert(0, str(DITTO))
os.chdir(DITTO)
import core.utils  # noqa: F401,E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("ditto")


def blend_np(mask, warped, rgb, out):
    m = mask[..., None]
    out[:] = np.clip(m * warped + (1 - m) * rgb, 0, 255).astype(np.uint8)


stub = types.ModuleType("core.utils.blend")
stub.blend_images_cy = blend_np
sys.modules["core.utils.blend"] = stub
from stream_pipeline_online import StreamSDK  # noqa: E402
from core.atomic_components.putback import PutBackNumpy  # noqa: E402


class Engine:
    def __init__(self):
        t = time.perf_counter()
        self.lock = threading.Lock()
        # Keep vendor configs untouched and write a short streaming-context copy.
        with open(DITTO / "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "rb") as f:
            cfg = pickle.load(f)
        cfg["default_kwargs"].update(online_mode=True, sampling_timesteps=5, max_size=1280)
        cfg_path = ROOT / "results/ditto_service_cfg.pkl"
        with open(cfg_path, "wb") as f:
            pickle.dump(cfg, f)
        self.sdk = StreamSDK(str(cfg_path), str(DITTO / "checkpoints/ditto_pytorch"))
        source = str(ROOT / "assets/face/tabi_still_ai_mirror.jpg")
        self.sdk.setup(source, str(ROOT / "results/ditto_service_unused.mp4"), online_mode=True,
                       sampling_timesteps=5, max_size=1280, overlap_v2=60)
        self.sdk.setup_Nd(N_d=1000000)
        # I drive the stages synchronously rather than using Ditto's MP4 writer threads.
        self.sdk.stop_event.set()
        self.sdk.audio2motion_queue.put(None)
        self.sdk.motion_stitch_queue.put(None)
        for thread in self.sdk.thread_list:
            thread.join()
        self.idle = self.sdk.source_info["img_rgb_lst"][0].copy()
        self.putback = PutBackNumpy()
        self.warp = self.sdk.warp_f3d.warp_net.model.eval().half()
        self.decoder = self.sdk.decode_f3d.decoder.model.eval().half()
        torch.backends.cudnn.benchmark = True
        self.warp = torch.compile(self.warp, mode="reduce-overhead")
        self.decoder = torch.compile(self.decoder, mode="reduce-overhead")
        self.audio_seq = []
        self.feature_buffer = np.zeros((0, self.sdk.wav2feat.feat_dim), dtype=np.float32)
        self.res_kp_seq = None
        self.valid_start = None
        self.local_idx = 0
        self.global_idx = 0
        self.generated_idx = 0
        self.audio_input_started = False
        self.audio_input_tail = np.zeros(0, dtype=np.float32)
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.pending_motions = []
        self.pending_frame = None
        self.latest_frame = self.idle.copy()
        self.items = collections.deque()
        self.pcm_tail = np.zeros(0, dtype=np.float32)
        self.active = False
        self.last_push = 0.0
        self.flushing = False
        self.started = threading.Event()
        log.info("Ditto models loaded in %.1fs; compiling fixed-size renderer", time.perf_counter() - t)
        t = time.perf_counter()
        with torch.inference_mode():
            f = torch.zeros(5, 32, 16, 64, 64, device="cuda", dtype=torch.float16)
            kp = torch.zeros(5, 21, 3, device="cuda", dtype=torch.float16)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                feats = self.warp(f, kp, kp).clone()
                torch.compiler.cudagraph_mark_step_begin()
                self.decoder(feats)
        torch.cuda.synchronize()
        self.started.set()
        log.info("Ditto ready; compile+warmup %.1fs", time.perf_counter() - t)

    def _features(self, chunk, queue_audio=True):
        audio = np.concatenate([np.zeros(1920, dtype=np.float32), chunk]) if not self.audio_input_started else chunk
        self.audio_input_started = True
        audio = np.concatenate([self.audio_input_tail, audio])
        motions = []
        for i in range(0, len(audio) - 3199, 3200):
            part = audio[i:i + 6480]
            if len(part) < 6480:
                part = np.pad(part, (0, 6480 - len(part)))
            self.feature_buffer = np.concatenate([self.feature_buffer, self.sdk.wav2feat(part)])
            valid_len = self.sdk.audio2motion.valid_clip_len
            seq_len = self.sdk.audio2motion.seq_frames
            if self.feature_buffer.shape[0] < valid_len:
                continue
            self.sdk.audio_feat = np.concatenate([self.sdk.audio_feat, self.feature_buffer])
            self.feature_buffer = np.zeros((0, self.sdk.wav2feat.feat_dim), dtype=np.float32)
            aud_feat = self.sdk.audio_feat[self.local_idx:self.local_idx + seq_len]
            if len(aud_feat) < seq_len:
                break
            cond = self.sdk.condition_handler(aud_feat, self.global_idx + self.sdk.cond_idx_start)[None]
            self.res_kp_seq = self.sdk.audio2motion(cond, self.res_kp_seq)
            if self.valid_start is None:
                self.valid_start = self.res_kp_seq.shape[1] - self.sdk.audio2motion.fuse_length
                d0 = self.sdk.audio2motion.cvt_fmt(self.res_kp_seq[:1])[0]
                self.sdk.motion_stitch.d0 = d0
            else:
                valid = self.res_kp_seq[:, self.valid_start:self.valid_start + valid_len]
                motions.extend(self.sdk.audio2motion.cvt_fmt(valid))
                self.valid_start += valid_len
            self.local_idx += valid_len
            self.global_idx += valid_len
            if self.res_kp_seq.shape[1] > seq_len * 2:
                cut = self.res_kp_seq.shape[1] - seq_len * 2
                self.res_kp_seq = self.res_kp_seq[:, cut:]
                self.valid_start -= cut
            if len(self.sdk.audio_feat) > seq_len * 2:
                cut = len(self.sdk.audio_feat) - seq_len * 2
                self.sdk.audio_feat = self.sdk.audio_feat[cut:]
                self.local_idx -= cut
            if self.local_idx >= len(self.sdk.audio_feat):
                break
        self.audio_input_tail = audio[(len(audio) // 3200) * 3200:]
        self.pending_motions.extend(motions)
        if queue_audio:
            self.pending_audio = np.concatenate([self.pending_audio, chunk])

    def _pop_audio(self, n):
        if len(self.pending_audio) >= n:
            out = self.pending_audio[:n]
            self.pending_audio = self.pending_audio[n:]
        else:
            out = np.pad(self.pending_audio, (0, n - len(self.pending_audio)))
            self.pending_audio = np.zeros(0, dtype=np.float32)
        return (out.clip(-1, 1) * 32767).astype(np.int16)

    def _render_batch(self, pairs):
        # Always infer five faces: changing shape would retrigger compilation.
        source = self.sdk.source_info
        full = pairs + [pairs[-1]] * (5 - len(pairs))
        t_a = time.perf_counter()
        with torch.inference_mode():
            # The second motion in each pair drives the held 80 ms face.
            stitched = [self.sdk.motion_stitch(source["x_s_info_lst"][0], pair[1]) for pair in full]
            f = torch.from_numpy(np.concatenate([source["f_s_lst"][0]] * 5)).cuda().half()
            ks = torch.from_numpy(np.concatenate([s[0] for s in stitched])).cuda().half()
            kd = torch.from_numpy(np.concatenate([s[1] for s in stitched])).cuda().half()
            torch.compiler.cudagraph_mark_step_begin()
            feats = self.warp(f, ks, kd).clone()
            torch.compiler.cudagraph_mark_step_begin()
            images = self.decoder(feats).float().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        t_b = time.perf_counter()
        for image, (_, _, pcm) in zip(images, pairs):
            frame = self.putback(source["img_rgb_lst"][0], image, source["M_c2o_lst"][0])
            self.items.append((frame, pcm))
        t_c = time.perf_counter()
        log.info("Ditto batch of %d: stitch+gpu %.0f ms, putback %.0f ms, queued %d",
                 len(pairs), (t_b - t_a) * 1000, (t_c - t_b) * 1000, len(self.items))

    def push(self, wav_bytes):
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        with self.lock:
            t = time.perf_counter()
            self._features(pcm)
            log.info("Ditto features for %.2fs audio: %.0f ms (%d motions pending)",
                     len(pcm) / 16000, (time.perf_counter() - t) * 1000, len(self.pending_motions))
            self.pending_audio = np.concatenate([self.pending_audio, pcm])
            self._push_pairs()
        self.last_push = time.perf_counter()

    def _push_pairs(self):
        while len(self.pending_motions) >= 2 and len(self.pending_audio) > 0:
            batch = []
            while len(batch) < 5 and len(self.pending_motions) >= 2 and len(self.pending_audio) > 0:
                a, b = self.pending_motions.pop(0), self.pending_motions.pop(0)
                batch.append((a, b, self._pop_audio(1280)))
            if batch:
                self._render_batch(batch)

    def flush_tail(self):
        with self.lock:
            if self.flushing or len(self.pending_audio) == 0:
                return
            self.flushing = True
            try:
                # HuBERT and the causal motion model need look-ahead. Feed it without
                # adding speech to the playback queue, then retain only real audio.
                self._features(np.zeros(32000, dtype=np.float32), queue_audio=False)
                self._push_pairs()
                self.pending_audio = np.zeros(0, dtype=np.float32)
                self.pending_motions.clear()
                self.pcm_tail = np.zeros(0, dtype=np.float32)
            finally:
                self.flushing = False

    def interrupt(self):
        with self.lock:
            self.items.clear()
            self.pending_audio = np.zeros(0, dtype=np.float32)
            self.pending_motions.clear()
            self.latest_frame = self.idle.copy()
            self.active = False


engine = None
pcs = set()
# CUDA graphs from torch.compile are thread-local, so every GPU call runs on this one thread.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


async def on_gpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(GPU, fn, *args)


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.pts = 0
        self.start = None
        self.current = np.zeros(0, dtype=np.int16)

    async def recv(self):
        if self.start is None:
            self.start = time.perf_counter()
        target = self.start + self.pts / 16000
        await asyncio.sleep(max(0, target - time.perf_counter()))
        if len(self.current) < 320:
            if engine.items and (engine.active or len(engine.items) >= 8):
                if not engine.active:
                    engine.active = True
                    log.info("Ditto audio start")
                frame, pcm = engine.items.popleft()
                engine.latest_frame = frame
                self.current = pcm
            else:
                self.current = np.zeros(1280, dtype=np.int16)
                if engine.active:
                    log.info("Ditto audio underrun/end")
                engine.active = False
        piece, self.current = self.current[:320], self.current[320:]
        frame = av.AudioFrame.from_ndarray(piece[None, :], format="s16", layout="mono")
        frame.sample_rate = 16000
        frame.pts = self.pts
        frame.time_base = Fraction(1, 16000)
        self.pts += 320
        return frame


class VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.pts = 0
        self.start = None

    async def recv(self):
        if self.start is None:
            self.start = time.perf_counter()
        target = self.start + self.pts / 25
        await asyncio.sleep(max(0, target - time.perf_counter()))
        frame = av.VideoFrame.from_ndarray(engine.latest_frame, format="rgb24")
        frame.pts = self.pts
        frame.time_base = Fraction(1, 25)
        self.pts += 1
        return frame


app = FastAPI()


@app.on_event("startup")
async def startup():
    global engine
    engine = await on_gpu(Engine)


@app.on_event("shutdown")
async def shutdown():
    for pc in list(pcs):
        await pc.close()


@app.get("/health")
async def health():
    return {"code": 0, "ready": engine is not None and engine.started.is_set()}


@app.post("/offer")
async def offer(request: Request):
    body = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)
    pc.addTrack(VideoTrack())
    pc.addTrack(AudioTrack())

    @pc.on("connectionstatechange")
    async def state():
        log.info("Ditto WebRTC %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(RTCSessionDescription(body["sdp"], body["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"code": 0, "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": "0"}


@app.post("/humanaudio")
async def humanaudio(sessionid: str = Form(...), file: UploadFile = File(...)):
    await on_gpu(engine.push, await file.read())
    return {"code": 0, "msg": "ok"}


@app.post("/is_speaking")
async def is_speaking(request: Request):
    if len(engine.pending_audio) > 0 and not engine.items and time.perf_counter() - engine.last_push > 0.8:
        await on_gpu(engine.flush_tail)
    active = engine.active or len(engine.items) > 0 or len(engine.pending_audio) > 0
    return {"code": 0, "data": active}


@app.post("/interrupt_talk")
async def interrupt(request: Request):
    engine.interrupt()
    return {"code": 0, "msg": "ok"}
