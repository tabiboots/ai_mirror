"""Standalone Wav2Lip speed and memory benchmark, no server involved.

Run from vendor/LiveTalking so `avatars` and `models/` resolve:
  $env:PYTHONUTF8=1; ..\\..\\.venv\\Scripts\\python ..\\..\\scripts\\bench_wav2lip.py [batch sizes...]
Mirrors LipReal.inference_batch: 6-channel 256x256 image + 80x16 mel -> model -> cpu numpy.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from avatars.wav2lip.models import Wav2Lip  # noqa: E402

REPS = 8
RES = 256
dev = torch.device("cuda")


def main(batch_sizes: list[int]) -> None:
    model = Wav2Lip()
    ckpt = torch.load("models/wav2lip.pth", map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()})
    model = model.to(dev).eval()
    print(f"after load: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            with torch.no_grad():
                for i in range(REPS + 1):
                    img = torch.FloatTensor(np.random.rand(bs, 6, RES, RES)).to(dev)
                    mel = torch.FloatTensor(np.random.rand(bs, 1, 80, 16)).to(dev)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pred = model(mel, img)
                    pred.cpu().numpy()
                    torch.cuda.synchronize()
                    if i > 0:
                        times.append(time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError as e:
            print(f"batch {bs:2}: OUT OF MEMORY ({str(e)[:60]})")
            continue
        t = float(np.median(times))
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"batch {bs:2}: {t * 1000:6.0f} ms  = {bs / t:6.1f} fps  peak {peak:.2f} GiB")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [4, 8, 16])
