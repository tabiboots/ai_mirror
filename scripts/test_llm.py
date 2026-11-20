"""Stream three chat turns with growing history; report time to first token and first sentence.

Run from the repo root:  .venv\\Scripts\\python -m scripts.test_llm
"""
import asyncio
import time

import httpx

from server import config, llm, net
from server.sentence import SentenceSplitter

TURNS = [
    "Hi, who am I talking to?",
    "What do you do all day?",
    "That sounds a bit strange. Does it bother you?",
]


async def main() -> None:
    print(f"LLM_MODEL={config.LLM_MODEL}  base={config.LLM_BASE_URL}")
    history: list[dict] = []
    for user in TURNS:
        splitter = SentenceSplitter()
        t0 = time.perf_counter()
        first_tok = first_sent = None
        reply = ""
        try:
            async for delta in llm.stream_reply(history, user):
                now = time.perf_counter() - t0
                if first_tok is None:
                    first_tok = now
                reply += delta
                if splitter.feed(delta) and first_sent is None:
                    first_sent = now
        except httpx.HTTPStatusError as e:
            print(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
            break
        total = time.perf_counter() - t0
        if first_sent is None and splitter.flush():
            first_sent = total
        print(f"user: {user}")
        print(f"bot : {reply}")
        print(
            f"  history={len(history)} msgs  first token {first_tok:.2f}s  "
            f"first sentence {first_sent:.2f}s  total {total:.2f}s\n"
        )
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]

    print("cancel mid-stream (1 call):")
    got: list[str] = []

    async def consume() -> None:
        async for d in llm.stream_reply([], "Tell me a long story about your childhood."):
            got.append(d)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(consume())
    while not got and not task.done():
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
        print("  finished before cancel")
    except asyncio.CancelledError:
        print(f"  cancelled after {len(got)} delta(s): CancelledError only")
    except Exception as e:
        print(f"  UNEXPECTED {type(e).__name__}: {e}")
    r = await net.get_client().get(config.LLM_BASE_URL + "/models", headers={"Authorization": f"Bearer {config.LLM_API_KEY}"})
    print(f"  shared client still usable after cancel: HTTP {r.status_code}")
    await net.aclose()


if __name__ == "__main__":
    asyncio.run(main())
