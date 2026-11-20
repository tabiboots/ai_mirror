import json
from pathlib import Path
from typing import AsyncIterator

from . import config, net

FALLBACK_PERSONA = (
    "You are an AI version of the person whose voice and face you use. "
    "You are talking with a visitor in a live video call. Be warm, curious and direct."
)

RULES = (
    "Rules for every reply: you are speaking out loud in a live call. "
    "Use short spoken sentences, 1 to 3 sentences per turn unless asked for more. "
    "No markdown, no lists, no emoji, no stage directions. "
    "Never claim to be a human. If asked, say you are an AI version of the user."
)


def system_prompt() -> str:
    parts = []
    persona = Path(config.PERSONA_PATH)
    parts.append(persona.read_text(encoding="utf-8").strip() if persona.is_file() else FALLBACK_PERSONA)
    samples = Path(config.SAMPLES_PATH)
    if samples.is_file():
        parts.append("Examples of how the person talks:\n" + samples.read_text(encoding="utf-8").strip())
    parts.append(RULES)
    return "\n\n".join(parts)


async def stream_reply(history: list[dict], user_text: str) -> AsyncIterator[str]:
    """Yield text deltas for the reply to user_text. history is prior {"role", "content"} turns.

    Cancelling the consuming task closes the HTTP stream.
    """
    if not config.LLM_MODEL:
        raise RuntimeError("Set LLM_MODEL in .env")
    if not config.LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY or GROQ_API_KEY in .env")
    messages = [
        {"role": "system", "content": system_prompt()},
        *history,
        {"role": "user", "content": user_text},
    ]
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 300,
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    async for line in net.stream("POST", url, lines=True, headers=headers, json=body):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            choices = json.loads(payload).get("choices") or []
        except json.JSONDecodeError:
            continue
        if not choices:
            continue
        delta = (choices[0].get("delta") or {}).get("content")
        if delta:
            yield delta
