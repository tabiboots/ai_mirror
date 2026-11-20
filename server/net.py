import asyncio
import random
from typing import AsyncIterator

import httpx

MAX_ATTEMPTS = 3
BACKOFF_START = 0.3
RETRY_AFTER_CAP = 10.0

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Shared client so TLS connections are reused between calls."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))
    return _client


async def aclose() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _retryable_status(code: int) -> bool:
    return code == 429 or code >= 500


def _delay(attempt: int, response: httpx.Response | None) -> float:
    d = BACKOFF_START * (2**attempt) * random.uniform(0.5, 1.5)
    if response is not None:
        ra = response.headers.get("retry-after")
        if ra:
            try:
                d = max(d, min(float(ra), RETRY_AFTER_CAP))
            except ValueError:
                pass
    return d


async def request(
    method: str, url: str, *, client: httpx.AsyncClient | None = None, **kwargs
) -> httpx.Response:
    """Send a request. Retries 429, 5xx and transport errors. Raises HTTPStatusError otherwise."""
    client = client or get_client()
    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        try:
            r = await client.request(method, url, **kwargs)
        except httpx.TransportError:
            if last:
                raise
            await asyncio.sleep(_delay(attempt, None))
            continue
        if _retryable_status(r.status_code) and not last:
            await asyncio.sleep(_delay(attempt, r))
            continue
        r.raise_for_status()
        return r
    raise AssertionError("unreachable")


async def stream(
    method: str,
    url: str,
    *,
    lines: bool = False,
    client: httpx.AsyncClient | None = None,
    **kwargs,
) -> AsyncIterator[bytes | str]:
    """Yield response bytes (or text lines). Retries only until the first item has been yielded.

    Cancelling the consumer closes the HTTP stream. Wrap the generator in
    contextlib.aclosing() if you may stop iterating early without cancelling.
    """
    client = client or get_client()
    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        started = False
        try:
            async with client.stream(method, url, **kwargs) as r:
                if _retryable_status(r.status_code) and not last:
                    await r.aread()
                    await asyncio.sleep(_delay(attempt, r))
                    continue
                if r.status_code >= 400:
                    await r.aread()
                    r.raise_for_status()
                it = r.aiter_lines() if lines else r.aiter_bytes()
                async for item in it:
                    started = True
                    yield item
                return
        except httpx.TransportError:
            if started or last:
                raise
            await asyncio.sleep(_delay(attempt, None))
