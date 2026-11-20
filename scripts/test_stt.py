"""Transcribe the Fish test clips with Groq Whisper; report latency and word error rate.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_stt
"""
import asyncio
import re
import subprocess
import time
from pathlib import Path

import httpx

from server import config, net, stt

CLIPS = {
    "line0": "Hey, it's me. Well, a version of me.",
    "line1": "I'm not sure I'd say it that way, but I get what you mean.",
    "line2": "Can you say that again? I think I missed the last part.",
}
DIR = Path("results/fish")


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("\u2019", "'")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def to_16k(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )


async def run(name: str, ref: str, path: Path) -> None:
    audio = path.read_bytes()
    t0 = time.perf_counter()
    try:
        text = await stt.transcribe(audio)
    except httpx.HTTPStatusError as e:
        print(f"{name}: HTTP {e.response.status_code}: {e.response.text[:300]}")
        return
    dt = time.perf_counter() - t0
    print(f"{name:10} {path.stat().st_size / 1024:6.1f} KB  {dt:5.2f}s  WER {wer(ref, text):.2f}  {text!r}")


async def main() -> None:
    print(f"STT_MODEL={config.STT_MODEL}")
    for name, ref in CLIPS.items():
        await run(f"{name}@44k", ref, DIR / f"{name}.wav")
    tmp = Path("results/stt")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, ref in CLIPS.items():
        dst = tmp / f"{name}_16k.wav"
        to_16k(DIR / f"{name}.wav", dst)
        await run(f"{name}@16k", ref, dst)
    await net.aclose()


if __name__ == "__main__":
    asyncio.run(main())
