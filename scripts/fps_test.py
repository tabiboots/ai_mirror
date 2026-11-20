"""Headless LiveTalking check: open a WebRTC session, play audio clips, count received video frames.

Run from the repo root with LiveTalking already running:
  .venv\\Scripts\\python -m scripts.fps_test [wav ...]
Server-side numbers ("actual avg infer fps") are in results/livetalking.log.
"""
import asyncio
import sys
import time
from pathlib import Path

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription

from server import config


async def main(wavs: list[str]) -> None:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    stamps: list[float] = []

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def drain():
            try:
                while True:
                    await track.recv()
                    stamps.append(time.perf_counter())
            except Exception:
                pass

        asyncio.ensure_future(drain())

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            f"{config.LIVETALKING_URL}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )
        r.raise_for_status()
        ans = r.json()
        sessionid = ans["sessionid"]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        print("session", sessionid)

        await asyncio.sleep(4)
        idle_n = len(stamps)
        print(f"idle: {idle_n} frames in 4 s")

        for wav in wavs:
            data = Path(wav).read_bytes()
            before = len(stamps)
            t0 = time.perf_counter()
            rr = await http.post(
                f"{config.LIVETALKING_URL}/humanaudio",
                data={"sessionid": sessionid},
                files={"file": (Path(wav).name, data, "audio/wav")},
            )
            print(wav, rr.status_code, rr.text[:80])
            async def speaking() -> bool:
                sp = await http.post(f"{config.LIVETALKING_URL}/is_speaking", json={"sessionid": sessionid})
                return bool(sp.json().get("data"))

            # wait for speech to start, then for it to stay finished for 1 s
            while not await speaking():
                await asyncio.sleep(0.1)
                if time.perf_counter() - t0 > 10:
                    break
            quiet_since = None
            while time.perf_counter() - t0 < 60:
                if await speaking():
                    quiet_since = None
                elif quiet_since is None:
                    quiet_since = time.perf_counter()
                elif time.perf_counter() - quiet_since > 1.0:
                    break
                await asyncio.sleep(0.1)
            dt = time.perf_counter() - t0
            n = len(stamps) - before
            print(f"  {n} frames received over {dt:.1f} s = {n / dt:.1f} fps (client side)")
            await asyncio.sleep(1)

    await pc.close()


if __name__ == "__main__":
    files = sys.argv[1:] or [f"results/fish/line{i}.wav" for i in range(3)]
    asyncio.run(main(files))
