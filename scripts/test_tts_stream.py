"""Test Fish streaming, connection reuse, pipelining, cancellation and the retry helper.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_tts_stream
Real Fish calls: 3 (reuse) + 3 (pipeline) + 1 (cancel) = 7. Retry tests use a fake transport.
"""
import asyncio
import contextlib
import time
import wave
from pathlib import Path

import httpx

from server import net, tts
from server.sentence import SentenceSplitter

LLM_TEXT = (
    "Hey, it's me. Well, a version of me. "
    "I'm not sure I'd say it that way, but I get what you mean. "
    "Can you say that again? I think I missed the last part."
)
DELTA_CHARS = 4
DELTA_GAP = 0.025
KNOWN = ["line0", "line1", "line2"]


def wav_seconds(path: Path) -> float:
    # Fish's WAV header has a bogus frame count, so derive duration from the file size.
    return (path.stat().st_size - 44) / (tts.PCM_RATE * tts.PCM_WIDTH * tts.PCM_CHANNELS)


async def fake_llm():
    for i in range(0, len(LLM_TEXT), DELTA_CHARS):
        await asyncio.sleep(DELTA_GAP)
        yield LLM_TEXT[i : i + DELTA_CHARS]


async def collect(text: str, t0: float, out: dict) -> None:
    out["start"] = time.perf_counter() - t0
    chunks = []
    async for c in tts.synth_stream(text):
        if "first" not in out:
            out["first"] = time.perf_counter() - t0
        chunks.append(c)
    out["done"] = time.perf_counter() - t0
    out["pcm"] = b"".join(chunks)
    out["chunks"] = len(chunks)


async def test_reuse() -> None:
    print("== connection reuse (same short sentence, 3 real calls)")
    text = "Hey, it's me."

    async def ttfb(client_label: str) -> None:
        t0 = time.perf_counter()
        async with contextlib.aclosing(tts.synth_stream(text)) as g:
            async for _ in g:
                print(f"  {client_label:28} first byte {time.perf_counter() - t0:5.2f}s")
                break

    for label in ("fresh client (cold TLS)", "same client again (warm)", "fresh client (cold TLS)"):
        if label.startswith("fresh"):
            await net.aclose()
        await ttfb(label)

    print("  connect cost alone (GET https://api.fish.audio/, no TTS call):")
    for label in ("cold", "warm", "cold", "warm"):
        if label == "cold":
            await net.aclose()
        t0 = time.perf_counter()
        r = await net.get_client().get("https://api.fish.audio/")
        print(f"    {label:5} HTTP {r.status_code} {time.perf_counter() - t0:5.2f}s")


async def test_pipeline() -> None:
    print("\n== pipeline: fake LLM -> splitter -> concurrent synth_stream (3 real calls)")
    await net.aclose()
    await tts.warm()
    splitter = SentenceSplitter()
    jobs: list[dict] = []
    tasks: list[asyncio.Task] = []
    t0 = time.perf_counter()

    def launch(sentence: str) -> None:
        job = {"text": sentence, "emitted": time.perf_counter() - t0}
        jobs.append(job)
        tasks.append(asyncio.create_task(collect(sentence, t0, job)))

    async for delta in fake_llm():
        for s in splitter.feed(delta):
            launch(s)
    for s in splitter.flush():
        launch(s)
    llm_done = time.perf_counter() - t0
    await asyncio.gather(*tasks)

    out = Path("results/tts_stream")
    out.mkdir(parents=True, exist_ok=True)
    print(f"  fake LLM stream finished at {llm_done:.2f}s")
    print(f"  {'#':2} {'emitted':>8} {'ttfb':>6} {'first@':>7} {'done@':>7} {'synth':>6} {'audio':>6}  text")
    for i, j in enumerate(jobs):
        dur = tts.pcm_seconds(j["pcm"])
        (out / f"s{i}.wav").write_bytes(tts.pcm_to_wav(j["pcm"]))
        j["dur"] = dur
        print(
            f"  {i:<2} {j['emitted']:7.2f}s {j['first'] - j['start']:5.2f}s {j['first']:6.2f}s "
            f"{j['done']:6.2f}s {j['done'] - j['start']:5.2f}s {dur:5.2f}s  {j['text']!r} ({j['chunks']} chunks)"
        )

    print("  PCM duration check against the known WAV clips of the same text:")
    for i, name in enumerate(KNOWN[: len(jobs)]):
        ref = wav_seconds(Path(f"results/fish/{name}.wav"))
        print(f"    sentence {i}: pcm {jobs[i]['dur']:.2f}s vs clip {name}.wav {ref:.2f}s")

    j0, j1 = jobs[0], jobs[1]
    print(f"  time to first audio from first delta: {j0['first']:.2f}s")
    play_start = j0["first"]
    need_s1 = play_start + j0["dur"]
    seq_s1_ready = j0["done"] + (j1["first"] - j1["start"])
    print(f"  playback of s0 would end at {need_s1:.2f}s; s1 first byte arrived at {j1['first']:.2f}s "
          f"({'before' if j1['first'] <= need_s1 else 'AFTER'} it, slack {need_s1 - j1['first']:+.2f}s)")
    print(f"  s1 first byte if started only after s0 finished: ~{seq_s1_ready:.2f}s; pipelined: {j1['first']:.2f}s; "
          f"hidden {seq_s1_ready - j1['first']:.2f}s")


async def test_cancel() -> None:
    print("\n== cancellation mid-stream (1 real call)")
    seen = []

    async def consumer() -> None:
        async for c in tts.synth_stream("Can you say that again? I think I missed the last part."):
            if not seen:
                print(f"  first chunk: {len(c)} bytes, starts with {c[:4]!r} (b'RIFF' would mean a WAV header)")
            seen.append(len(c))
            await asyncio.sleep(0.05)

    task = asyncio.create_task(consumer())
    while not seen:
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
        print("  task finished without cancel (stream ended first)")
    except asyncio.CancelledError:
        print(f"  cancelled after {len(seen)} chunk(s), CancelledError only, no other exception")
    except Exception as e:
        print(f"  UNEXPECTED {type(e).__name__}: {e}")
    t0 = time.perf_counter()
    r = await net.get_client().get("https://api.fish.audio/")
    print(f"  shared client still usable after cancel: HTTP {r.status_code} in {time.perf_counter() - t0:.2f}s")


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_retry() -> None:
    print("\n== retry helper (fake transport, no network)")
    net.BACKOFF_START = 0.01

    def counted(script):
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            step = script[min(calls["n"], len(script) - 1)]
            calls["n"] += 1
            if isinstance(step, Exception):
                raise step
            return step

        return calls, mock_client(handler)

    async def check(name, coro, calls, expect_calls, expect):
        try:
            res = await coro
            got = f"ok:{res}"
        except Exception as e:
            got = f"raise:{type(e).__name__}" + (f"({e.response.status_code})" if isinstance(e, httpx.HTTPStatusError) else "")
        status = "PASS" if calls["n"] == expect_calls and got == expect else "FAIL"
        print(f"  {status}  {name}: calls={calls['n']} (want {expect_calls}) result={got} (want {expect})")

    r429 = lambda: httpx.Response(429, headers={"retry-after": "0.05"})
    ok = lambda: httpx.Response(200, content=b"hi")

    async def req(c):
        return (await net.request("POST", "http://x/", client=c)).text

    async def strm(c, lines=False):
        return [x async for x in net.stream("POST", "http://x/", client=c, lines=lines)]

    calls, c = counted([r429(), ok()])
    t = time.perf_counter()
    await check("429 then 200", req(c), calls, 2, "ok:hi")
    print(f"        (waited {time.perf_counter() - t:.2f}s, Retry-After 0.05 honored if >= 0.05)")
    calls, c = counted([httpx.Response(503), httpx.Response(502), ok()])
    await check("503, 502, then 200", req(c), calls, 3, "ok:hi")
    calls, c = counted([httpx.Response(500)])
    await check("500 forever -> give up after 3", req(c), calls, 3, "raise:HTTPStatusError(500)")
    calls, c = counted([httpx.Response(400)])
    await check("400 is not retried", req(c), calls, 1, "raise:HTTPStatusError(400)")
    calls, c = counted([httpx.Response(401)])
    await check("401 is not retried", req(c), calls, 1, "raise:HTTPStatusError(401)")
    calls, c = counted([httpx.ConnectError("boom"), ok()])
    await check("transport error then 200", req(c), calls, 2, "ok:hi")
    calls, c = counted([httpx.ConnectError("boom")])
    await check("transport error forever", req(c), calls, 3, "raise:ConnectError")
    calls, c = counted([httpx.Response(429), ok()])
    await check("stream: 429 before first byte, then ok", strm(c), calls, 2, "ok:[b'hi']")
    calls, c = counted([httpx.Response(404)])
    await check("stream: 404 not retried", strm(c), calls, 1, "raise:HTTPStatusError(404)")

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"abc"
            raise httpx.ReadError("dropped mid-stream")

    calls, c = counted([httpx.Response(200, stream=Broken())])
    await check("stream: error after first byte is NOT retried", strm(c), calls, 1, "raise:ReadError")

    class Slow(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            for _ in range(100):
                yield b"x"
                await asyncio.sleep(0.01)

        async def aclose(self):
            Slow.closed = True

    calls, c = counted([httpx.Response(200, stream=Slow())])

    async def early_exit():
        async with contextlib.aclosing(net.stream("POST", "http://x/", client=c)) as g:
            async for _ in g:
                break

    await early_exit()
    print(f"  {'PASS' if Slow.closed else 'FAIL'}  stream: early exit with aclosing closes the response")

    Slow.closed = False
    calls, c = counted([httpx.Response(200, stream=Slow())])
    got = []

    async def consume():
        async for x in net.stream("POST", "http://x/", client=c):
            got.append(x)

    task = asyncio.create_task(consume())
    while not got:
        await asyncio.sleep(0.005)
    task.cancel()
    try:
        await task
        res = "no CancelledError"
    except asyncio.CancelledError:
        res = "CancelledError only"
    except Exception as e:
        res = f"UNEXPECTED {type(e).__name__}"
    print(f"  {'PASS' if Slow.closed and res == 'CancelledError only' else 'FAIL'}  stream: task.cancel() "
          f"-> {res}, response closed={Slow.closed}")


async def main() -> None:
    await test_retry()
    await test_reuse()
    await test_pipeline()
    await test_cancel()
    await net.aclose()


if __name__ == "__main__":
    asyncio.run(main())
