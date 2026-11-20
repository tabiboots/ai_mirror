"""Headless WebRTC smoke test for the experimental Ditto service on port 8011."""
import asyncio
import io
import time
import wave
from pathlib import Path

import cv2
import httpx
import librosa
import numpy as np
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription

ROOT = Path(__file__).resolve().parent.parent


async def main():
    async with httpx.AsyncClient(timeout=180) as http:
        print("waiting for Ditto", flush=True)
        for _ in range(180):
            try:
                r = await http.get("http://127.0.0.1:8011/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Ditto service did not start")

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        frames, audio_samples = [], []

        @pc.on("track")
        def on_track(track):
            async def receive():
                try:
                    while True:
                        f = await track.recv()
                        if track.kind == "video":
                            frames.append(f.to_ndarray(format="bgr24"))
                        else:
                            audio_samples.append(f.to_ndarray())
                except Exception:
                    pass
            asyncio.create_task(receive())

        await pc.setLocalDescription(await pc.createOffer())
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        answer = (await http.post("http://127.0.0.1:8011/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
        })).json()
        await pc.setRemoteDescription(RTCSessionDescription(answer["sdp"], answer["type"]))
        await asyncio.sleep(0.5)

        pcm, _ = librosa.load(str(ROOT / "results/tts_stream/s1.wav"), sr=16000)
        raw = (pcm.clip(-1, 1) * 32767).astype(np.int16).tobytes()
        first = int(0.44 * 16000) * 2
        chunks = [raw[i:i + first if i == 0 else i + 32000] for i in range(0, len(raw), 32000) if i]
        chunks.insert(0, raw[:first])
        t0 = time.perf_counter()
        for i, chunk in enumerate(chunks):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(chunk)
            r = await http.post("http://127.0.0.1:8011/humanaudio", data={"sessionid": "0"},
                                files={"file": ("line.wav", buf.getvalue(), "audio/wav")})
            print(i, r.status_code, r.text, f"{time.perf_counter() - t0:.2f}s", flush=True)
        await asyncio.sleep(10)
        await pc.close()

    print("video frames", len(frames), "audio samples", sum(x.size for x in audio_samples), flush=True)
    if not frames:
        return
    out_dir = ROOT / "results/ditto_live_frames"
    out_dir.mkdir(exist_ok=True)
    out = cv2.VideoWriter(str(ROOT / "results/ditto_live.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25,
                          (frames[0].shape[1], frames[0].shape[0]))
    changes = 0
    for i, frame in enumerate(frames):
        out.write(frame)
        if i and np.mean(np.abs(frame.astype(float) - frames[i - 1].astype(float))) > 0.2:
            changes += 1
        if i % 25 == 0:
            cv2.imwrite(str(out_dir / f"f{i:03d}.png"), frame)
    out.release()
    print("changed display frames", changes, flush=True)


asyncio.run(main())
