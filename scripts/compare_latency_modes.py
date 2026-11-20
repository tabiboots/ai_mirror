"""Synthesize the same sentences at different Fish latency modes and save them for listening.

Run from the repo root:  .venv\\Scripts\\python -m scripts.compare_latency_modes [normal balanced low]
Each mode costs one Fish call per sentence. Default modes: normal balanced.
Writes results/fish/compare/<mode>_<n>.wav
"""
import asyncio
import sys
import time
from pathlib import Path

import httpx

from server import net, tts

SENTENCES = [
    "I'm not sure I'd say it that way, but I get what you mean.",
    "Can you say that again? I think I missed the last part.",
]


async def one(mode: str, n: int, text: str, out: Path) -> tuple[str, float, float, float] | None:
    t0 = time.perf_counter()
    first = None
    chunks = []
    try:
        async for c in tts.synth_stream(text, latency=mode):
            if first is None:
                first = time.perf_counter() - t0
            chunks.append(c)
    except httpx.HTTPStatusError as e:
        print(f"{mode}_{n}: HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    total = time.perf_counter() - t0
    pcm = b"".join(chunks)
    path = out / f"{mode}_{n}.wav"
    path.write_bytes(tts.pcm_to_wav(pcm))
    return str(path), first, total, tts.pcm_seconds(pcm)


async def main() -> None:
    modes = sys.argv[1:] or ["normal", "balanced"]
    out = Path("results/fish/compare")
    out.mkdir(parents=True, exist_ok=True)
    await tts.warm()
    rows = []
    for mode in modes:
        for n, text in enumerate(SENTENCES):
            row = await one(mode, n, text, out)
            if row:
                rows.append(row)
    print(f"{'file':42} {'first byte':>10} {'total':>7} {'audio':>7}")
    for path, first, total, dur in rows:
        print(f"{path:42} {first:9.2f}s {total:6.2f}s {dur:6.2f}s")
    for n, text in enumerate(SENTENCES):
        print(f"sentence {n}: {text!r}")
    await net.aclose()


if __name__ == "__main__":
    asyncio.run(main())
