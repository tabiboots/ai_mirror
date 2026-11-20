# Asset capture checklist

Raw files stay in this folder and are git-ignored. Only you, only your own face and voice.

## 1. Voice sample -> `assets/voice/`

| Item | Spec |
|---|---|
| File | `voice_ref.wav` (mono, 44.1 kHz or 48 kHz, 16/24-bit). Keep the original too. |
| Length | 20-30 s main clip. Also cut a 10 s clip, `voice_ref_10s.wav`, to compare. |
| Content | Natural, varied speech in your normal talking pace: a few statements, a question, one excited line. Not reading flatly. |
| Room | Quiet, soft furnishings, no echo. Phone 10-15 cm from your mouth, or a headset mic. |
| Avoid | Music, fans, keyboard noise, other voices, clipping, long silences, heavy noise reduction. |
| Transcript | Write the exact words to `voice_ref.txt`. Some engines need it, and it improves cloning. |

Record 3 takes. Keep the cleanest.

## 2. Face video -> `assets/face/`

| Item | Spec |
|---|---|
| File | `face_ref.mp4`, 25 fps, 720p or 1080p, H.264. |
| Length | 30-60 s. |
| Framing | Head and shoulders, face centered, looking straight at the lens, camera at eye level. |
| Expression | Neutral, mouth mostly closed, blink naturally, small head movements only. |
| Lighting | Soft, even, from the front. No hard side shadows. |
| Background | Plain and static. |
| Avoid | Glasses glare, hair across the mouth, hands near the face, heavy head turns, talking. |

Also grab one still of the same setup, `face_still.png`, for the fallback path.

Optional, for a later "idle" loop: 10-15 s of you sitting and listening, `idle_loop.mp4`.

## 3. Persona text -> `assets/persona/`

- `persona.md`: how you talk, topics, opinions, what the agent must never claim.
- `samples.txt`: 20-50 real lines you have said or written. These become the LLM's few-shot examples.
