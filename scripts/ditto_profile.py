"""Offline Ditto run with per-stage timers (PyTorch checkpoints, no TensorRT).

Run from vendor/ditto-talkinghead with .venv-ditto:
  ..\\..\\.venv-ditto\\Scripts\\python.exe ..\\..\\scripts\\ditto_profile.py --steps 10
"""
import argparse
import collections
import math
import os
import sys
import time
import types

import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import core.utils  # noqa: E402,F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _blend_images_np(mask, frame_warped, frame_rgb, out):
    m = mask[..., None]
    out[:] = np.clip(m * frame_warped + (1.0 - m) * frame_rgb, 0, 255).astype(np.uint8)


_stub = types.ModuleType("core.utils.blend")
_stub.blend_images_cy = _blend_images_np
sys.modules["core.utils.blend"] = _stub

from stream_pipeline_offline import StreamSDK  # noqa: E402

T = collections.defaultdict(float)
N = collections.defaultdict(int)


class Timed:
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def __call__(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = self._inner(*a, **k)
        torch.cuda.synchronize()
        T[self._name] += time.perf_counter() - t
        N[self._name] += 1
        return r

    def __getattr__(self, item):
        return getattr(self._inner, item)


def timed_method(obj, attr, name):
    fn = getattr(obj, attr)

    def wrap(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        T[name] += time.perf_counter() - t
        N[name] += 1
        return r

    setattr(obj, attr, wrap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=os.path.join(ROOT, "results", "tts_stream", "s1.wav"))
    ap.add_argument("--source", default=os.path.join(ROOT, "assets", "face", "tabi_still_ai_mirror.jpg"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ditto_profile.mp4"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max_size", type=int, default=1280)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--data_root", default="./checkpoints/ditto_pytorch")
    ap.add_argument("--cfg", default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    args = ap.parse_args()

    sdk = StreamSDK(args.cfg, args.data_root)
    timed_method(sdk.wav2feat, "wav2feat", "hubert")
    sdk.audio2motion.lmdm = Timed(sdk.audio2motion.lmdm, "lmdm_diffusion")
    sdk.motion_stitch = Timed(sdk.motion_stitch, "motion_stitch")
    sdk.warp_f3d = Timed(sdk.warp_f3d, "warp")
    sdk.decode_f3d = Timed(sdk.decode_f3d, "decode")
    sdk.putback = Timed(sdk.putback, "putback_blend")

    audio, _ = librosa.core.load(args.audio, sr=16000)
    dur = len(audio) / 16000
    num_f = math.ceil(dur * 25)

    for run in range(args.runs):
        T.clear()
        N.clear()
        out = args.out if run == args.runs - 1 else args.out + f".warm{run}.mp4"
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        sdk.setup(args.source, out, sampling_timesteps=args.steps, max_size=args.max_size)
        sdk.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
        t_setup = time.perf_counter() - t

        t = time.perf_counter()
        aud_feat = sdk.wav2feat.wav2feat(audio)
        sdk.audio2motion_queue.put(aud_feat)
        sdk.close()
        t_gen = time.perf_counter() - t
        frames = sdk.writer_pbar.n
        print(
            f"[run {run}] audio={dur:.2f}s expected_frames={num_f} written={frames} "
            f"avatar_setup={t_setup:.2f}s generate={t_gen:.2f}s "
            f"RTF={t_gen / dur:.2f} fps={frames / t_gen:.1f} "
            f"vram_peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )
        for k in T:
            per = T[k] / max(N[k], 1) * 1000
            print(f"    {k:16s} total={T[k]:6.2f}s  calls={N[k]:4d}  per_call={per:7.1f} ms")
    print("out:", out)


if __name__ == "__main__":
    main()
