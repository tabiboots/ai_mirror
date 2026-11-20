"""Standalone MuseTalk speed and memory benchmark, no server involved.

Run from vendor/LiveTalking so `avatars` and `models/` resolve:
  $env:PYTHONUTF8=1; ..\\..\\.venv\\Scripts\\python ..\\..\\scripts\\bench_musetalk.py [batch sizes...]
Mirrors MuseReal.inference_batch: pe -> unet -> vae.decode_latents.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from avatars.musetalk.utils.utils import load_all_model  # noqa: E402

REPS = 6
dev = torch.device("cuda")


def sync():
    torch.cuda.synchronize()


def main(batch_sizes: list[int]) -> None:
    free, total = torch.cuda.mem_get_info()
    print(f"gpu free/total before load: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")

    vae, unet, pe = load_all_model()
    timesteps = torch.tensor([0], device=dev)
    pe = pe.half().to(dev)
    vae.vae = vae.vae.half().to(dev)
    unet.model = unet.model.half().to(dev)
    print(f"after load: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        whisper = torch.from_numpy(np.ones((bs, 50, 384), dtype=np.uint8))
        latent = torch.ones(bs, 8, 32, 32)
        t_unet, t_vae = [], []
        try:
            with torch.no_grad():
                for i in range(REPS + 1):
                    sync()
                    t0 = time.perf_counter()
                    a = pe(whisper.to(device=dev, dtype=unet.model.dtype))
                    pred = unet.model(latent.to(dev, dtype=unet.model.dtype), timesteps,
                                      encoder_hidden_states=a).sample
                    sync()
                    t1 = time.perf_counter()
                    vae.decode_latents(pred)
                    sync()
                    t2 = time.perf_counter()
                    if i > 0:
                        t_unet.append(t1 - t0)
                        t_vae.append(t2 - t1)
        except torch.cuda.OutOfMemoryError as e:
            print(f"batch {bs:2}: OUT OF MEMORY ({str(e)[:60]})")
            continue
        u, v = float(np.median(t_unet)), float(np.median(t_vae))
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"batch {bs:2}: unet {u * 1000:6.0f} ms  vae {v * 1000:6.0f} ms  "
              f"total {(u + v) * 1000:6.0f} ms  = {bs / (u + v):5.1f} fps  peak {peak:.2f} GiB")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [4, 8, 16])
