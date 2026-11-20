"""Generate a few test lines with your Fish voice and report time to audio.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_fish
Writes results/fish/*.wav
"""
import asyncio
import time
from pathlib import Path

import httpx

from server import tts

LINES = [
    "Hey, it's me. Well, a version of me.",
    "I'm not sure I'd say it that way, but I get what you mean.",
    "Can you say that again? I think I missed the last part.",
]


async def main() -> None:
    out = Path("results/fish")
    out.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(LINES):
        t0 = time.perf_counter()
        try:
            audio = await tts.synth(line)
        except httpx.HTTPStatusError as e:
            print(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
            return
        except RuntimeError as e:
            print(e)
            return
        dt = time.perf_counter() - t0
        path = out / f"line{i}.wav"
        path.write_bytes(audio)
        print(f"{dt:5.2f}s  {len(audio) / 1024:7.1f} KB  {path}  {line!r}")


if __name__ == "__main__":
    asyncio.run(main())
