from . import config, net

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


async def transcribe(
    audio: bytes,
    language: str | None = "en",
    filename: str = "utterance.wav",
    content_type: str = "audio/wav",
) -> str:
    """Transcribe one short utterance with Groq Whisper. Accepts WAV, webm/opus, etc."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY in .env")
    data = {
        "model": config.STT_MODEL,
        "response_format": "json",
        "temperature": "0",
    }
    if language:
        data["language"] = language
    r = await net.request(
        "POST",
        f"{GROQ_BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        data=data,
        files={"file": (filename, audio, content_type)},
    )
    return r.json().get("text", "").strip()
