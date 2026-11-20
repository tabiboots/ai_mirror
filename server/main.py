"""AI_Mirror server. Run from the repo root:

  .venv\\Scripts\\python -m uvicorn server.main:app --port 8000

LiveTalking must already be running (see PLAN.md). Open http://localhost:8000
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import net, tts
from .bridge import LiveTalking
from .turn import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("mirror")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bridge = LiveTalking()
    yield
    await app.state.bridge.aclose()
    await net.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/offer")
async def offer(request: Request):
    """Pass the browser's WebRTC offer to LiveTalking. The answer carries the sessionid."""
    body = await request.json()
    cands = [l.split()[4] for l in body["sdp"].splitlines() if l.startswith("a=candidate:")]
    log.info("offer has %d candidates: %s", len(cands), cands)
    try:
        return await request.app.state.bridge.offer(body["sdp"], body["type"])
    except Exception as e:
        log.error("offer failed: %s", e)
        raise HTTPException(502, f"LiveTalking: {e}")


@app.websocket("/ws")
async def ws(websocket: WebSocket, sessionid: str):
    await websocket.accept()

    async def send(msg: dict) -> None:
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            pass

    session = Session(sessionid, websocket.app.state.bridge, send)
    await tts.warm()
    await send({"type": "status", "state": "listening"})
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.start_turn(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") in ("speech_start", "interrupt"):
                    await session.barge_in()
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
