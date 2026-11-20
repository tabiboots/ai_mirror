import io
import math
import wave
from typing import AsyncIterator

import numpy as np
from scipy.signal import resample_poly

from . import config, net

FISH_URL = "https://api.fish.audio/v1/tts"
# "s2.1-pro-free" is the free developer tier per Fish's docs. Change to "s2.1-pro" for the paid model.
FISH_MODEL = "s2.1-pro-free"

# Fish "pcm" output: signed 16-bit little-endian, mono, 44.1 kHz.
PCM_RATE = 44100
PCM_WIDTH = 2
PCM_CHANNELS = 1

LIVETALKING_RATE = 16000


def _request(text: str, fmt: str, latency: str) -> dict:
    if not config.FISH_API_KEY or not config.FISH_REFERENCE_ID:
        raise RuntimeError("Set FISH_API_KEY and FISH_REFERENCE_ID in .env")
    return {
        "url": FISH_URL,
        "headers": {
            "Authorization": f"Bearer {config.FISH_API_KEY}",
            "Content-Type": "application/json",
            "model": FISH_MODEL,
        },
        "json": {
            "text": text,
            "reference_id": config.FISH_REFERENCE_ID,
            "format": fmt,
            "latency": latency,
            "normalize": True,
        },
    }


async def warm() -> None:
    """Open the TLS connection to Fish ahead of the first sentence. No TTS call is made."""
    try:
        await net.get_client().get("https://api.fish.audio/")
    except Exception:
        pass


async def synth(text: str, fmt: str = "wav", latency: str = "balanced") -> bytes:
    """Return the full audio for one sentence from Fish Audio using the saved voice."""
    req = _request(text, fmt, latency)
    r = await net.request("POST", req.pop("url"), **req)
    return r.content


async def synth_stream(
    text: str, fmt: str = "pcm", latency: str = "balanced"
) -> AsyncIterator[bytes]:
    """Yield audio chunks as Fish produces them.

    For fmt="pcm" every chunk has an even length, so it never splits a 16-bit sample.
    """
    req = _request(text, fmt, latency)
    carry = b""
    async for chunk in net.stream("POST", req.pop("url"), **req):
        if fmt == "pcm":
            chunk = carry + chunk
            if len(chunk) % PCM_WIDTH:
                chunk, carry = chunk[:-1], chunk[-1:]
            else:
                carry = b""
            if not chunk:
                continue
        yield chunk


def pcm_to_wav(pcm: bytes, rate: int = PCM_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(PCM_WIDTH)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_to_wav16k(pcm: bytes) -> bytes:
    """Resample Fish PCM to a 16 kHz WAV, the rate LiveTalking plays at.

    Doing it here keeps the resampling off LiveTalking's event loop.
    """
    g = math.gcd(LIVETALKING_RATE, PCM_RATE)
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    y = resample_poly(x, LIVETALKING_RATE // g, PCM_RATE // g)
    return pcm_to_wav(np.clip(y, -32768, 32767).astype("<i2").tobytes(), rate=LIVETALKING_RATE)


def pcm_seconds(pcm: bytes, rate: int = PCM_RATE) -> float:
    return len(pcm) / (rate * PCM_WIDTH * PCM_CHANNELS)
