"""Download the MuseTalk v1.5 weights that LiveTalking loads at runtime.

Run from the repo root:  .venv\\Scripts\\python scripts\\download_weights.py
Skips files that already exist. SyncNet and DWPose are not fetched: LiveTalking does not load them.
"""
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = Path("vendor/LiveTalking/models")

HF_FILES = [
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "musetalkV15"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "musetalkV15"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "sd-vae"),
    ("openai/whisper-tiny", "config.json", "whisper"),
    ("openai/whisper-tiny", "pytorch_model.bin", "whisper"),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper"),
]

RESNET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
FACE_PARSE_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


def fetch_hf(repo: str, filename: str, dest_dir: str) -> None:
    out = MODELS / dest_dir / Path(filename).name
    if out.exists():
        print(f"have  {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copyfile(cached, out)
    print(f"got   {out}")


def main() -> int:
    failed = []
    for repo, filename, dest in HF_FILES:
        try:
            fetch_hf(repo, filename, dest)
        except Exception as e:
            print(f"FAIL  {repo}/{filename}: {e}")
            failed.append(filename)

    fp = MODELS / "face-parse-bisent"
    fp.mkdir(parents=True, exist_ok=True)

    resnet = fp / "resnet18-5c106cde.pth"
    if not resnet.exists():
        try:
            urllib.request.urlretrieve(RESNET_URL, resnet)
            print(f"got   {resnet}")
        except Exception as e:
            print(f"FAIL  resnet18: {e}")
            failed.append("resnet18")

    bisent = fp / "79999_iter.pth"
    if not bisent.exists():
        try:
            import gdown

            gdown.download(id=FACE_PARSE_GDRIVE_ID, output=str(bisent), quiet=False)
        except Exception as e:
            print(f"FAIL  79999_iter.pth (Google Drive): {e}")
        if not bisent.exists():
            failed.append("79999_iter.pth")

    if failed:
        print("\nMissing:", ", ".join(failed))
        return 1
    print("\nAll weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
