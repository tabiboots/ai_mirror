import asyncio
import logging
import time
from typing import Awaitable, Callable

from . import llm, stt, tts
from .bridge import LiveTalking
from .sentence import SentenceSplitter

log = logging.getLogger("mirror.turn")

# 882 samples at 44.1 kHz is exactly 20 ms, which resamples to 320 samples at 16 kHz.
# LiveTalking plays 20 ms frames and drops the tail of a file that is not a whole number of them.
UNIT = 882 * tts.PCM_WIDTH
FIRST_CHUNK = 22 * UNIT  # about 0.44 s of audio, posted as soon as it exists
CHUNK = 50 * UNIT  # about 1 s
MAX_HISTORY = 16
TTS_CONCURRENCY = 2
END_QUIET_S = 0.6


class Session:
    """One browser connection: conversation history and the turn that is currently running."""

    def __init__(
        self,
        sessionid: str,
        bridge: LiveTalking,
        send: Callable[[dict], Awaitable[None]],
    ):
        self.sessionid = sessionid
        self.bridge = bridge
        self.send = send
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.sem = asyncio.Semaphore(TTS_CONCURRENCY)

    async def start_turn(self, wav: bytes) -> None:
        await self.cancel()
        self.task = asyncio.create_task(self._turn(wav, time.perf_counter()))

    async def barge_in(self) -> None:
        """The user started talking. Stop the avatar and drop the running turn."""
        if self.task is None or self.task.done():
            return
        await self.cancel()
        await self._state("listening")

    async def cancel(self) -> None:
        task, self.task = self.task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                await self.bridge.interrupt(self.sessionid)
            except Exception as e:
                log.warning("interrupt failed: %s", e)

    async def close(self) -> None:
        task, self.task = self.task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _state(self, state: str) -> None:
        await self.send({"type": "status", "state": state})

    async def _synth(self, sentence: str, out: asyncio.Queue, first_limit: int, marks: Callable) -> None:
        """Fetch one sentence from Fish and put it in `out` as PCM chunks. None marks the end."""
        try:
            async with self.sem:
                buf = bytearray()
                limit = first_limit
                async for piece in tts.synth_stream(sentence):
                    marks("first_tts_byte")
                    buf += piece
                    while len(buf) >= limit:
                        out.put_nowait(bytes(buf[:limit]))
                        del buf[:limit]
                        limit = CHUNK
                if buf:
                    pad = -len(buf) % UNIT
                    out.put_nowait(bytes(buf) + b"\x00" * pad)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("tts failed for %r: %s", sentence, e)
            await self.send({"type": "error", "where": "tts", "message": str(e)})
        finally:
            out.put_nowait(None)

    async def _produce(self, text: str, order: asyncio.Queue, marks: Callable, tasks: list, reply: list) -> None:
        """Stream the LLM, split into sentences and start synthesis for each as soon as it exists."""
        splitter = SentenceSplitter()

        async def launch(sentence: str) -> None:
            marks("first_sentence")
            reply.append(sentence)
            out: asyncio.Queue = asyncio.Queue()
            first = FIRST_CHUNK if len(tasks) == 0 else CHUNK
            tasks.append(asyncio.create_task(self._synth(sentence, out, first, marks)))
            order.put_nowait(out)
            await self.send({"type": "agent_text", "text": sentence})

        try:
            async for delta in llm.stream_reply(self.history, text):
                marks("first_llm_token")
                for s in splitter.feed(delta):
                    await launch(s)
            for s in splitter.flush():
                await launch(s)
            marks("llm_done")
        finally:
            order.put_nowait(None)

    async def _wait_playback_end(self, audio_s: float) -> None:
        deadline = time.perf_counter() + audio_s + 5
        quiet_since = None
        while time.perf_counter() < deadline:
            speaking = await self.bridge.is_speaking(self.sessionid)
            now = time.perf_counter()
            if speaking:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= END_QUIET_S:
                return
            await asyncio.sleep(0.15)

    async def _turn(self, wav: bytes, t0: float) -> None:
        marks_t: dict[str, float] = {}

        def marks(name: str) -> None:
            marks_t.setdefault(name, round(time.perf_counter() - t0, 3))

        tasks: list[asyncio.Task] = []
        producer: asyncio.Task | None = None
        reply: list[str] = []
        user_text = ""
        spoken_s = 0.0
        try:
            await self._state("thinking")
            user_text = await stt.transcribe(wav)
            marks("stt_done")
            if len(user_text) < 2:
                await self._state("listening")
                return
            await self.send({"type": "user_text", "text": user_text})

            order: asyncio.Queue = asyncio.Queue()
            producer = asyncio.create_task(self._produce(user_text, order, marks, tasks, reply))
            speaking = False
            while True:
                out = await order.get()
                if out is None:
                    break
                while True:
                    chunk = await out.get()
                    if chunk is None:
                        break
                    wav16 = await asyncio.to_thread(tts.pcm_to_wav16k, chunk)
                    await self.bridge.play_audio(self.sessionid, wav16)
                    spoken_s += tts.pcm_seconds(chunk)
                    marks("first_audio_posted")
                    if not speaking:
                        speaking = True
                        await self._state("speaking")
            await producer
            marks("all_audio_posted")
            if speaking:
                await self._wait_playback_end(spoken_s)
            marks("playback_end")
            await self.send({"type": "timings", "marks": marks_t, "audio_s": round(spoken_s, 2)})
            await self._state("listening")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("turn failed")
            await self.send({"type": "error", "where": "turn", "message": str(e)})
            await self._state("listening")
        finally:
            for t in [producer, *tasks]:
                if t is not None and not t.done():
                    t.cancel()
            await asyncio.gather(*[t for t in [producer, *tasks] if t is not None], return_exceptions=True)
            if user_text and reply:
                self.history += [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": " ".join(reply)},
                ]
                self.history = self.history[-MAX_HISTORY:]
