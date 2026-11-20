"""Offline test: does input loudness change how far Wav2Lip opens the mouth?

Run from vendor/LiveTalking:
  $env:PYTHONUTF8=1; ..\\..\\.venv\\Scripts\\python ..\\..\\scripts\\gain_test.py [wav]
Writes results/gain/strip_<gain>.png (mouth crops over time) and prints a mouth-opening score.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, ".")
from avatars.wav2lip import audio  # noqa: E402
from avatars.wav2lip.models import Wav2Lip  # noqa: E402

dev = torch.device("cuda")
OUT = Path("../../results/gain")
OUT.mkdir(parents=True, exist_ok=True)


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def load_wav16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x


def silence_mouth(model, img_t, mouth) -> np.ndarray:
    mel = audio.melspectrogram(np.zeros(16000, dtype=np.float32))
    m = torch.FloatTensor(mel[:, :16]).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = model(m, img_t)
    return (p[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[mouth].astype(np.float32)


def main(wav: str) -> None:
    model = Wav2Lip()
    ck = torch.load("models/wav2lip.pth", map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["state_dict"].items()})
    model = model.to(dev).eval()

    faces = [cv2.imread(f) for f in sorted(glob.glob("data/avatars/tabi_wav2lip/face_imgs/*.png"))]
    x = load_wav16k(wav)
    print(f"{wav}: {len(x) / 16000:.1f}s  rms {db(np.sqrt(np.mean(x**2))):.1f} dBFS  peak {db(np.abs(x).max()):.1f} dBFS")

    # one fixed face frame, so only the audio changes between runs
    face = faces[len(faces) // 3]
    masked = face.copy()
    masked[face.shape[0] // 2 :] = 0
    img = np.concatenate([masked, face], axis=2) / 255.0
    img_t = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)

    gains = [("x1", 1.0), ("x2", 2.0), ("x4", 4.0), ("x8", 8.0), ("peak0.9", None)]
    h = face.shape[0]
    mouth = slice(int(h * 0.62), int(h * 0.92)), slice(int(h * 0.25), int(h * 0.75))
    for name, g in gains:
        y = x * g if g else x * (0.9 / np.abs(x).max())
        y = np.clip(y, -1, 1)
        mel = audio.melspectrogram(y)
        n = int(mel.shape[1] / 3.2) - 5
        crops, opens = [], []
        for i in range(0, n, 2):
            s = int(i * 3.2)
            chunk = mel[:, s : s + 16]
            if chunk.shape[1] < 16:
                break
            m = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = model(m, img_t)
            p = (p[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            m_crop = p[mouth]
            crops.append(m_crop)
            opens.append(m_crop.astype(np.float32))
        # mean abs pixel change vs the same face driven by silence = how much the audio moves the mouth
        sil = silence_mouth(model, img_t, mouth)
        o = np.array([np.abs(c - sil).mean() for c in opens])
        print(f"{name:8s} rms {db(np.sqrt(np.mean(y**2))):6.1f} dBFS  diff-from-silence mean {o.mean():.2f}  p90 {np.percentile(o, 90):.2f}  max {o.max():.2f}")
        pick = crops[:: max(1, len(crops) // 12)][:12]
        cv2.imwrite(str(OUT / f"strip_{name}.png"), np.hstack(pick))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../results/tts_stream/s1.wav")
