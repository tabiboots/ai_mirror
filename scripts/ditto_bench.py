"""Offline Ditto benchmark on the PyTorch checkpoints (no TensorRT).

Run from vendor/ditto-talkinghead with .venv-ditto:
  ..\\..\\.venv-ditto\\Scripts\\python.exe ..\\..\\scripts\\ditto_bench.py --steps 10
"""
import argparse
import math
import os
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

# Ditto's Cython blend needs MSVC; stub it with an equivalent NumPy blend.
import types  # noqa: E402

import core.utils  # noqa: E402,F401


def _blend_images_np(mask, frame_warped, frame_rgb, out):
    m = mask[..., None]
    out[:] = np.clip(m * frame_warped + (1.0 - m) * frame_rgb, 0, 255).astype(np.uint8)


_stub = types.ModuleType("core.utils.blend")
_stub.blend_images_cy = _blend_images_np
sys.modules["core.utils.blend"] = _stub

from stream_pipeline_online import StreamSDK  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=os.path.join(ROOT, "results", "tts_stream", "s1.wav"))
    ap.add_argument("--source", default=os.path.join(ROOT, "assets", "face", "tabi_still_ai_mirror.jpg"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ditto_bench.mp4"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max_size", type=int, default=1280)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--data_root", default="./checkpoints/ditto_pytorch")
    ap.add_argument("--cfg", default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    args = ap.parse_args()

    t0 = time.time()
    sdk = StreamSDK(
        args.cfg,
        args.data_root,
        default_kwargs={"online_mode": True, "sampling_timesteps": args.steps, "max_size": args.max_size},
    )
    print(f"[load] models: {time.time() - t0:.2f}s")

    audio, _ = librosa.core.load(args.audio, sr=16000)
    dur = len(audio) / 16000
    num_f = math.ceil(dur * 25)
    chunksize = (3, 5, 2)
    audio_p = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
    split_len = int(sum(chunksize) * 0.04 * 16000) + 80

    for run in range(args.runs):
        out = args.out if run == args.runs - 1 else args.out + f".warm{run}.mp4"
        t = time.time()
        sdk.setup(args.source, out, online_mode=True, sampling_timesteps=args.steps, max_size=args.max_size)
        sdk.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})
        t_setup = time.time() - t

        t = time.time()
        t_first = None
        for i in range(0, len(audio_p), chunksize[1] * 640):
            chunk = audio_p[i:i + split_len]
            if len(chunk) < split_len:
                chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
            sdk.run_chunk(chunk, chunksize)
            if t_first is None and sdk.writer_pbar.n > 0:
                t_first = time.time() - t
        t_feed = time.time() - t
        sdk.close()
        t_total = time.time() - t
        torch.cuda.synchronize()
        print(
            f"[run {run}] audio={dur:.2f}s frames={num_f} setup(avatar)={t_setup:.2f}s "
            f"feed={t_feed:.2f}s total={t_total:.2f}s RTF={t_total / dur:.2f} "
            f"first_frame={'%.2fs' % t_first if t_first else 'n/a'} "
            f"fps={num_f / t_total:.1f} vram_peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )
    print("out:", out)


if __name__ == "__main__":
    main()
