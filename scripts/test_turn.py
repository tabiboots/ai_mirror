"""One full conversation turn without video: user audio -> STT -> LLM -> sentences -> Fish PCM.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_turn
Writes results/turn/reply.wav
"""
import asyncio
import time
from pathlib import Path

from server import llm, net, stt, tts
from server.sentence import SentenceSplitter

USER_AUDIO = Path("results/fish/line1.wav")


async def main() -> None:
    marks: dict[str, float] = {}
    t0 = time.perf_counter()

    def mark(name: str) -> None:
        marks.setdefault(name, time.perf_counter() - t0)

    await tts.warm()
    t0 = time.perf_counter()

    user_text = await stt.transcribe(USER_AUDIO.read_bytes())
    mark("stt_done")
    print(f"user: {user_text!r}")

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    pcm_parts: dict[int, bytes] = {}

    async def speak(i: int, sentence: str) -> None:
        buf = []
        async for chunk in tts.synth_stream(sentence):
            if i == 0:
                mark("first_audio_byte")
            buf.append(chunk)
        pcm_parts[i] = b"".join(buf)
        mark(f"sentence{i}_audio_done")

    tasks = []
    splitter = SentenceSplitter()
    reply = ""

    def launch(sentence: str) -> None:
        mark("first_sentence")
        print(f"sentence {len(tasks)}: {sentence!r}")
        tasks.append(asyncio.create_task(speak(len(tasks), sentence)))

    async for delta in llm.stream_reply([], user_text):
        mark("first_llm_token")
        reply += delta
        for s in splitter.feed(delta):
            launch(s)
    mark("llm_done")
    for s in splitter.flush():
        launch(s)
    await asyncio.gather(*tasks)

    pcm = b"".join(pcm_parts[i] for i in sorted(pcm_parts))
    out = Path("results/turn")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reply.wav").write_bytes(tts.pcm_to_wav(pcm))
    print(f"reply: {reply!r}")
    print(f"saved results/turn/reply.wav  {tts.pcm_seconds(pcm):.2f}s of audio")
    for k, v in sorted(marks.items(), key=lambda kv: kv[1]):
        print(f"  {k:24} {v:5.2f}s")
    print(f"time to first audio byte (from end of user audio, no VAD/video): {marks['first_audio_byte']:.2f}s")
    await net.aclose()


if __name__ == "__main__":
    asyncio.run(main())
