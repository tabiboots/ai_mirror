"""Measure Fish Audio time-to-first-byte and total time, per format and latency mode.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_fish_stream
"""
import asyncio
import time

import httpx

from server import config
from server.tts import FISH_MODEL, FISH_URL

TEXT = "I'm not sure I'd say it that way, but I get what you mean."


async def one(fmt: str, latency: str) -> None:
    headers = {
        "Authorization": f"Bearer {config.FISH_API_KEY}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
    }
    body = {
        "text": TEXT,
        "reference_id": config.FISH_REFERENCE_ID,
        "format": fmt,
        "latency": latency,
        "normalize": True,
    }
    t0 = time.perf_counter()
    first = None
    total = 0
    chunks = 0
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", FISH_URL, headers=headers, json=body) as r:
            if r.status_code != 200:
                print(f"{fmt:4} {latency:9} HTTP {r.status_code}: {(await r.aread())[:200]!r}")
                return
            async for chunk in r.aiter_bytes():
                if first is None:
                    first = time.perf_counter() - t0
                total += len(chunk)
                chunks += 1
    dt = time.perf_counter() - t0
    print(f"{fmt:4} {latency:9} first byte {first:5.2f}s  total {dt:5.2f}s  {total / 1024:6.1f} KB  {chunks} chunks")


async def main() -> None:
    for fmt in ("pcm", "wav", "mp3"):
        for latency in ("normal", "balanced", "low"):
            await one(fmt, latency)


if __name__ == "__main__":
    asyncio.run(main())
