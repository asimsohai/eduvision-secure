"""Download the OpenCV face models and verify their SHA-256 checksums."""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, expected) in MODELS.items():
        path = MODELS_DIR / name
        if path.exists() and sha256(path) == expected:
            print(f"ok       {name}")
            continue
        print(f"download {name} ...")
        urllib.request.urlretrieve(url, path)
        if sha256(path) != expected:
            path.unlink()
            sys.exit(f"Checksum mismatch for {name}; file deleted. Do not use it.")
        print(f"verified {name}")


if __name__ == "__main__":
    main()
