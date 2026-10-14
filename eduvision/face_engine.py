"""Face detection (YuNet) and face embeddings (SFace), both built into OpenCV.

Each face becomes a 128-number vector (an "embedding"). Two photos of the same
person give vectors pointing in a similar direction, measured by cosine
similarity. A match needs similarity >= settings.match_threshold.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import settings


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    raw: np.ndarray                 # YuNet row (box + 5 landmarks + score)


class FaceEngine:
    def __init__(self):
        for path in (settings.detector_model, settings.recognizer_model):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model {path.name}. Run: python scripts/download_models.py"
                )
        self.detector = cv2.FaceDetectorYN.create(
            str(settings.detector_model), "", (320, 320),
            score_threshold=settings.detection_threshold,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(settings.recognizer_model), "")

    def detect(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        return [
            Face(box=tuple(int(v) for v in f[:4]), score=float(f[-1]), raw=f)
            for f in faces
        ]

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face.raw)
        vec = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return vec / np.linalg.norm(vec)  # unit length: dot product = cosine


def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Matcher:
    """Holds all enrolled embeddings in memory (loaded once, not per frame)."""

    def __init__(self, rows):
        self.labels = [(r["student_pk"], r["student_id"], r["name"]) for r in rows]
        self.matrix = (
            np.vstack([from_blob(r["embedding"]) for r in rows])
            if rows else np.empty((0, 128), np.float32)
        )

    def best_match(self, vec: np.ndarray, threshold: float | None = None):
        """Return (student_pk, student_id, name, similarity) or None."""
        threshold = settings.match_threshold if threshold is None else threshold
        if len(self.matrix) == 0:
            return None
        sims = self.matrix @ vec
        i = int(np.argmax(sims))
        if sims[i] < threshold:
            return None
        return (*self.labels[i], float(sims[i]))
