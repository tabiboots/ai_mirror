import httpx

from . import config


class LiveTalking:
    """Thin client for LiveTalking's HTTP routes (vendor/LiveTalking/server/routes.py)."""

    def __init__(self, base_url: str | None = None):
        self.base = (base_url or config.LIVETALKING_URL).rstrip("/")
        self.http = httpx.AsyncClient(timeout=30)

    async def _post_json(self, path: str, body: dict) -> dict:
        r = await self.http.post(f"{self.base}{path}", json=body)
        r.raise_for_status()
        data = r.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"{path}: {data.get('msg')}")
        return data

    async def offer(self, sdp: str, type_: str) -> dict:
        """Start a WebRTC session. Returns {"sdp", "type", "sessionid"}."""
        r = await self.http.post(f"{self.base}/offer", json={"sdp": sdp, "type": type_})
        r.raise_for_status()
        data = r.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"/offer: {data.get('msg')}")
        return data

    async def play_audio(self, sessionid: str, wav_bytes: bytes) -> None:
        r = await self.http.post(
            f"{self.base}/humanaudio",
            data={"sessionid": sessionid},
            files={"file": ("line.wav", wav_bytes, "audio/wav")},
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code", 0) != 0:
            raise RuntimeError(f"/humanaudio: {data.get('msg')}")

    async def interrupt(self, sessionid: str) -> None:
        await self._post_json("/interrupt_talk", {"sessionid": sessionid})

    async def is_speaking(self, sessionid: str) -> bool:
        data = await self._post_json("/is_speaking", {"sessionid": sessionid})
        return bool(data.get("data"))

    async def aclose(self) -> None:
        await self.http.aclose()
