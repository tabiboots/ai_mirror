const RATE = 16000;
const START_RMS = 0.02;       // speech starts above this
const START_RMS_BUSY = 0.05;  // higher while the avatar is talking, so its own audio does not interrupt it
const END_SILENCE_MS = 700;
const START_HOLD_MS = 150;
const PREROLL_MS = 300;
const MIN_SPEECH_MS = 350;
const MAX_UTTERANCE_S = 20;

const video = document.getElementById("video");
const audio = document.getElementById("audio");

let pc, ws, ctx, state = "idle";
let started = false;
let preroll = [];            // recent chunks while idle
let utter = null;            // chunks of the current utterance, or null
let loudMs = 0, quietMs = 0, utterMs = 0;

function waitForIce(pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const done = () => { pc.removeEventListener("icegatheringstatechange", check); resolve(); };
    const check = () => pc.iceGatheringState === "complete" && done();
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(done, 2000);
  });
}

async function connectVideo() {
  pc = new RTCPeerConnection();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.addEventListener("track", (e) => {
    if (e.track.kind === "video") video.srcObject = e.streams[0];
    else audio.srcObject = e.streams[0];
  });
  pc.addEventListener("connectionstatechange", () => {
    console.log("webrtc", pc.connectionState, "ice", pc.iceConnectionState);
  });
  await pc.setLocalDescription(await pc.createOffer());
  await waitForIce(pc);
  const r = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });
  if (!r.ok) throw new Error(`offer failed: ${r.status} ${await r.text()}`);
  const answer = await r.json();
  await pc.setRemoteDescription(answer);
  return answer.sessionid;
}

function connectSocket(sessionid) {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(`ws://${location.host}/ws?sessionid=${encodeURIComponent(sessionid)}`);
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error("websocket failed"));
    ws.onclose = () => { state = "disconnected"; ws = null; };
    ws.onmessage = (e) => onServerMessage(JSON.parse(e.data));
  });
}

function onServerMessage(m) {
  if (m.type === "status") state = m.state;
  else if (m.type === "error") console.error(m);
}

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  ctx = new AudioContext({ sampleRate: RATE });
  await ctx.audioWorklet.addModule("worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "capture");
  node.port.onmessage = (e) => onChunk(e.data);
  src.connect(node);
}

function rms(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

function onChunk(buf) {
  const ms = (buf.length / RATE) * 1000;
  const level = rms(buf);
  if (!ws) return;

  if (!utter) {
    preroll.push(buf);
    while (preroll.length * ms > PREROLL_MS) preroll.shift();
    const threshold = state === "speaking" ? START_RMS_BUSY : START_RMS;
    loudMs = level > threshold ? loudMs + ms : 0;
    if (loudMs >= START_HOLD_MS) beginUtterance();
    return;
  }

  utter.push(buf);
  utterMs += ms;
  quietMs = level > START_RMS ? 0 : quietMs + ms;
  if (quietMs >= END_SILENCE_MS || utterMs > MAX_UTTERANCE_S * 1000) endUtterance();
}

function beginUtterance() {
  if (utter) return;
  utter = [...preroll];
  preroll = [];
  utterMs = 0; quietMs = 0; loudMs = 0;
  ws.send(JSON.stringify({ type: "speech_start" }));
}

function endUtterance() {
  if (!utter) return;
  const chunks = utter;
  const speechMs = utterMs - quietMs;
  utter = null;
  if (speechMs < MIN_SPEECH_MS || !ws) return;
  ws.send(toWav(chunks));
  state = "thinking";
}

function toWav(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new DataView(new ArrayBuffer(44 + n * 2));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); out.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, RATE, true); out.setUint32(28, RATE * 2, true);
  out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  w(36, "data"); out.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++, o += 2) {
      out.setInt16(o, Math.max(-1, Math.min(1, c[i])) * 0x7fff, true);
    }
  }
  return out.buffer;
}

// Browsers require a user gesture for mic access and audio playback, so click the video to start.
video.addEventListener("click", async () => {
  if (started) return;
  started = true;
  state = "connecting";
  try {
    // Mic first: Chrome only puts real IPs in WebRTC candidates after mic permission is granted.
    await startMic();
    const sessionid = await connectVideo();
    await connectSocket(sessionid);
    video.classList.add("live");
  } catch (e) {
    console.error(e);
    state = "error";
    started = false;
  }
});
