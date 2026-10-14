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
    detection_threshold: float = float(os.getenv("DETECTION_THRESHOLD", "0.75"))
    max_failed_logins: int = int(os.getenv("MAX_FAILED_LOGINS", "5"))
    lockout_minutes: int = int(os.getenv("LOCKOUT_MINUTES", "15"))
    # off | blink | challenge (random head-turn challenge, the default)
    liveness_mode: str = (
        "off" if os.getenv("LIVENESS_ENABLED", "true").lower() == "false"
        else os.getenv("LIVENESS_MODE", "challenge").lower()
    )
    challenge_steps: int = int(os.getenv("CHALLENGE_STEPS", "3"))
    challenge_step_seconds: float = float(os.getenv("CHALLENGE_STEP_SECONDS", "6"))
    challenge_max_attempts: int = int(os.getenv("CHALLENGE_MAX_ATTEMPTS", "2"))
    challenge_lockout_seconds: float = float(os.getenv("CHALLENGE_LOCKOUT_SECONDS", "30"))
    challenge_hold_frames: int = int(os.getenv("CHALLENGE_HOLD_FRAMES", "2"))
    turn_threshold: float = float(os.getenv("TURN_THRESHOLD", "0.12"))
    challenge_flip: bool = os.getenv("CHALLENGE_FLIP", "false").lower() == "true"
    required_blinks: int = int(os.getenv("REQUIRED_BLINKS", "1"))
    blink_min_closed_frames: int = int(os.getenv("BLINK_MIN_CLOSED_FRAMES", "2"))
    blink_max_closed_frames: int = int(os.getenv("BLINK_MAX_CLOSED_FRAMES", "8"))

    @property
    def detector_model(self) -> Path:
        return self.models_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def recognizer_model(self) -> Path:
        return self.models_dir / "face_recognition_sface_2021dec.onnx"


settings = Settings()
