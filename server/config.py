import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


LIVETALKING_URL = _get("LIVETALKING_URL", "http://127.0.0.1:8010")

FISH_API_KEY = _get("FISH_API_KEY")
FISH_REFERENCE_ID = _get("FISH_REFERENCE_ID")

GROQ_API_KEY = _get("GROQ_API_KEY")
STT_MODEL = _get("STT_MODEL", "whisper-large-v3-turbo")

LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = _get("LLM_API_KEY") or GROQ_API_KEY
LLM_MODEL = _get("LLM_MODEL")

PERSONA_PATH = _get("PERSONA_PATH", "assets/persona/persona.md")
SAMPLES_PATH = _get("SAMPLES_PATH", "assets/persona/samples.txt")
