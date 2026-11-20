"""Micro-benchmark of Ditto's warp + decode networks under cheap optimisations.

Run from vendor/ditto-talkinghead with .venv-ditto.
"""
import copy
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import core.utils  # noqa: E402,F401

_stub = types.ModuleType("core.utils.blend")
_stub.blend_images_cy = lambda *a, **k: None
sys.modules["core.utils.blend"] = _stub

from core.models.decoder import Decoder  # noqa: E402
from core.models.warp_network import WarpNetwork  # noqa: E402

R = "./checkpoints/ditto_pytorch/models/"
warp = WarpNetwork(R + "warp_network.pth").model.eval()
dec = Decoder(R + "decoder.pth").model.eval()
dev = "cuda"


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def make_inputs(b, dtype):
    f = torch.randn(b, 32, 16, 64, 64, device=dev, dtype=dtype)
    ks = torch.randn(b, 21, 3, device=dev, dtype=dtype)
    kd = torch.randn(b, 21, 3, device=dev, dtype=dtype)
    return f, ks, kd


def run(label, warp_m, dec_m, b, dtype, autocast):
    f, ks, kd = make_inputs(b, dtype)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            wf = warp_m(f, ks, kd).clone()
        wf = wf.to(dtype) if dtype == torch.float16 else wf

        def w():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                return warp_m(f, ks, kd)

        def d():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                return dec_m(wf)

        tw, td = bench(w), bench(d)
    print(f"{label:34s} batch={b}  warp={tw / b:6.1f} ms/frame  decode={td / b:6.1f} ms/frame  total={(tw + td) / b:6.1f} ms/frame")


torch.backends.cudnn.benchmark = False
run("baseline eager autocast", warp, dec, 1, torch.float32, True)

torch.backends.cudnn.benchmark = True
run("+ cudnn.benchmark", warp, dec, 1, torch.float32, True)

wh, dh = copy.deepcopy(warp).half(), copy.deepcopy(dec).half()
run("+ pure fp16 weights", wh, dh, 1, torch.float16, False)
run("+ pure fp16, batch 5", wh, dh, 5, torch.float16, False)

wc = copy.deepcopy(wh)
dc = copy.deepcopy(dh).to(memory_format=torch.channels_last)
run("+ fp16 + channels_last decoder, b=5", wc, dc, 5, torch.float16, False)

if "--compile" in sys.argv:
    print("compiling fp16 models...", flush=True)
    t = time.perf_counter()
    wcomp = torch.compile(copy.deepcopy(wh), mode="reduce-overhead")
    dcomp = torch.compile(copy.deepcopy(dh), mode="reduce-overhead")
    run("+ torch.compile fp16, batch 5", wcomp, dcomp, 5, torch.float16, False)
    print("compile + benchmark wall", round(time.perf_counter() - t, 1), "s")

print("vram_peak", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
