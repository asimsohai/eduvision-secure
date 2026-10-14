"""Settings loaded from the environment (.env), never hard-coded in the code."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    db_path: Path = _path("DB_PATH", "data/attendance.db")
    models_dir: Path = _path("MODELS_DIR", "models")
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", "0.363"))
    detection_threshold: float = float(os.getenv("DETECTION_THRESHOLD", "0.9"))
    max_failed_logins: int = int(os.getenv("MAX_FAILED_LOGINS", "5"))
    lockout_minutes: int = int(os.getenv("LOCKOUT_MINUTES", "15"))

    @property
    def detector_model(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def recognizer_model(self) -> Path:
        return self.models_dir / "face_recognition_sface_2021dec.onnx"


settings = Settings()
