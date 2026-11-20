import re

_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
_ABBREV = ("mr.", "mrs.", "ms.", "dr.", "e.g.", "i.e.", "vs.", "etc.")


class SentenceSplitter:
    """Turns a stream of LLM text chunks into speakable sentences.

    The first sentence is released early (min_first chars) so TTS can start sooner.
    """

    def __init__(self, min_first: int = 20, min_rest: int = 40):
        self.buf = ""
        self.min_first = min_first
        self.min_rest = min_rest
        self.emitted = 0

    def _min(self) -> int:
        return self.min_first if self.emitted == 0 else self.min_rest

    def feed(self, chunk: str) -> list[str]:
        self.buf += chunk
        out: list[str] = []
        start = 0
        for m in _END.finditer(self.buf):
            candidate = self.buf[start : m.start()].strip()
            if candidate.lower().endswith(_ABBREV):
                continue
            if len(candidate) < self._min():
                continue
            out.append(self.buf[start : m.end()].strip())
            self.emitted += 1
            start = m.end()
        self.buf = self.buf[start:]
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        if rest:
            self.emitted += 1
            return [rest]
        return []
