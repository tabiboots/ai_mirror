"""12.5 fps Ditto: motion at 25 fps, fp16 warp+decode every 2nd frame.

Run from vendor/ditto-talkinghead using .venv-ditto.
"""
import argparse
import math
import os
import queue
import sys
import threading
import time
import types

import cv2
import imageio
import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import core.utils  # noqa: F401
stub = types.ModuleType("core.utils.blend")
stub.blend_images_cy = lambda *a: None
sys.modules["core.utils.blend"] = stub
from stream_pipeline_offline import StreamSDK  # noqa: E402
from core.atomic_components.putback import PutBackNumpy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=os.path.join(ROOT, "results", "tts_stream", "s1.wav"))
    ap.add_argument("--source", default=os.path.join(ROOT, "assets", "face", "tabi_still_ai_mirror.jpg"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ditto_12fps.mp4"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    sdk = StreamSDK(
        "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        "./checkpoints/ditto_pytorch",
        default_kwargs={"max_size": 1280, "sampling_timesteps": args.steps},
    )
    audio, _ = librosa.load(args.audio, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)
    duration = len(audio) / 16000

    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    t = time.perf_counter()
    sdk.setup(args.source, args.out + ".pipeline.mp4", max_size=1280, sampling_timesteps=args.steps)
    sdk.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
    # Stop all but the audio/motion stage; its output is consumed below.
    for thread in sdk.thread_list[1:]:
        sdk.stop_event.set()
    sdk.motion_stitch_queue.put(None)
    for thread in sdk.thread_list[1:]:
        thread.join()
    print(f"avatar setup: {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    aud_feat = sdk.wav2feat.wav2feat(audio)
    aud_cond_all = sdk.condition_handler(aud_feat, 0)
    seq_frames = sdk.audio2motion.seq_frames
    valid_clip_len = sdk.audio2motion.valid_clip_len
    res_kp_seq = None
    for idx in range(0, len(aud_cond_all), valid_clip_len):
        aud_cond = aud_cond_all[idx:idx + seq_frames][None]
        if aud_cond.shape[1] < seq_frames:
            pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
            aud_cond = np.concatenate([aud_cond, pad], 1)
        res_kp_seq = sdk.audio2motion(aud_cond, res_kp_seq)
    res_kp_seq = res_kp_seq[:, :len(aud_cond_all)]
    res_kp_seq = sdk.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])
    motions = sdk.audio2motion.cvt_fmt(res_kp_seq)
    print(f"audio + motion: {time.perf_counter() - t:.2f}s ({len(motions)} frames)")

    # Half weights let us keep tensors on the GPU and avoid autocast conversions.
    torch.backends.cudnn.benchmark = True
    warp = sdk.warp_f3d.warp_net.model.eval().half()
    decoder = sdk.decode_f3d.decoder.model.eval().half()
    if args.compile:
        warp = torch.compile(warp, mode="reduce-overhead")
        decoder = torch.compile(decoder, mode="reduce-overhead")
    putback = PutBackNumpy()
    source = sdk.source_info
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out + ".tmp.mp4", fps=25, codec="libx264", quality=7)
    render_times = []
    total_written = 0

    with torch.inference_mode():
        for start in range(0, len(motions), args.batch * 2):
            ids = list(range(start, min(start + args.batch * 2, len(motions)), 2))
            if not ids:
                continue
            stitched = [sdk.motion_stitch(source["x_s_info_lst"][0], motions[i]) for i in ids]
            f = np.concatenate([source["f_s_lst"][0]] * len(ids)).astype(np.float32)
            ks = np.concatenate([s[0] for s in stitched]).astype(np.float32)
            kd = np.concatenate([s[1] for s in stitched]).astype(np.float32)
            torch.cuda.synchronize()
            t = time.perf_counter()
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            f = torch.from_numpy(f).cuda().half()
            ks = torch.from_numpy(ks).cuda().half()
            kd = torch.from_numpy(kd).cuda().half()
            feats = warp(f, ks, kd).clone()
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            renders = decoder(feats).float().clamp(0, 1).mul(255).byte()
            renders = renders.permute(0, 2, 3, 1).cpu().numpy()
            torch.cuda.synchronize()
            render_times.append((time.perf_counter() - t, len(ids)))
            print(f"  batch {start // (args.batch * 2)}: {render_times[-1][0]:.2f}s for {len(ids)}; f16={feats.dtype}")

            for i, render in zip(ids, renders):
                frame = putback(source["img_rgb_lst"][0], render, source["M_c2o_lst"][0])
                writer.append_data(frame)  # 0,0,2,2,4,4 ... holds each face for 80 ms
                writer.append_data(frame)
                total_written += 2
    writer.close()
    wall = time.perf_counter() - t_all
    gpu_s = sum(t for t, _ in render_times)
    frames = sum(n for _, n in render_times)
    print(f"render: {gpu_s:.2f}s for {frames} unique faces; {gpu_s / frames * 1000:.1f} ms/face")
    print(f"total: {wall:.2f}s for {duration:.2f}s audio; RTF={wall / duration:.2f}; output frames={total_written}; peak VRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(args.out + ".tmp.mp4")


if __name__ == "__main__":
    main()
