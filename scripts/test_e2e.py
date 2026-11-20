"""Headless browser stand-in: WebRTC via the mirror server, WebSocket turn, count video frames.

Needs LiveTalking (8010) and the mirror server (8000) running. From the repo root:
  .venv\\Scripts\\python -m scripts.test_e2e [user_clip.wav]
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

BASE = os.environ.get("MIRROR_BASE", "127.0.0.1:8000")


async def main(clip: Path) -> None:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    stamps: list[float] = []

    @pc.on("track")
    def on_track(track):
        async def drain():
            try:
                while True:
                    await track.recv()
                    if track.kind == "video":
                        stamps.append(time.perf_counter())
            except Exception:
                pass

        asyncio.ensure_future(drain())

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(
            f"http://{BASE}/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )
        r.raise_for_status()
        ans = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    sid = ans["sessionid"]
    print("session", sid)
    await asyncio.sleep(2)

    async with websockets.connect(f"ws://{BASE}/ws?sessionid={sid}", max_size=None) as ws:
        t_send = time.perf_counter()
        await ws.send(json.dumps({"type": "speech_start"}))
        await ws.send(clip.read_bytes())
        speaking_at = None
        n0 = len(stamps)
        t_end = None
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                print("timeout waiting for server")
                break
            m = json.loads(raw)
            dt = time.perf_counter() - t_send
            if m["type"] == "status":
                print(f"{dt:6.2f}s status {m['state']}")
                if m["state"] == "speaking" and speaking_at is None:
                    speaking_at = time.perf_counter()
                    n0 = len(stamps)
                if m["state"] == "listening" and speaking_at is not None:
                    t_end = time.perf_counter()
            elif m["type"] == "timings":
                print("timings", json.dumps(m["marks"]), "audio_s", m["audio_s"])
                if t_end is None:
                    continue
            else:
                print(f"{dt:6.2f}s {m['type']}: {m.get('text') or m.get('message')}")
            if t_end is not None and m["type"] != "timings":
                break
        if speaking_at and t_end:
            n = len(stamps) - n0
            print(f"video: {n} frames in {t_end - speaking_at:.1f}s while speaking = {n / (t_end - speaking_at):.1f} fps")
    await pc.close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/stt/line1_16k.wav")))
