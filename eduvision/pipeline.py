"""The per-frame decision pipeline shared by live attendance and evaluation.

One person at a time (kiosk style):
  no face / several faces -> wait
  face not recognised     -> "unknown"
  recognised, not yet live -> "liveness" (blink or head-turn challenge pending)
  recognised and live      -> "accepted"
Liveness restarts whenever the person changes or leaves the camera.
"""
import time
from dataclasses import dataclass

from .face_engine import Face, FaceEngine, Matcher
from .liveness import make_checker

# Seconds without a recognised face before liveness restarts. Turned heads are
# sometimes missed by the detector for a moment, so this must not be too short.
SECONDS_BEFORE_RESET = 2.0


@dataclass
class FrameResult:
    status: str                  # none | multiple | unknown | liveness | accepted
    face: Face | None = None
    match: tuple | None = None   # (student_pk, student_id, name, similarity)
    prompt: str = ""             # instruction for the person (liveness)
    debug: str = ""              # live measurement, shown small on screen


class Pipeline:
    def __init__(self, engine: FaceEngine, matcher: Matcher, mode: str = "challenge"):
        self.engine = engine
        self.matcher = matcher
        self.mode = mode
        self.liveness = make_checker(mode)
        self._current_pk = None
        self._last_seen = None
        self.missing_frames = 0   # for evaluation diagnostics

    def reset(self) -> None:
        self.liveness.reset()
        self._current_pk = None
        self._last_seen = None

    def _face_missing(self) -> None:
        self.missing_frames += 1
        if self._last_seen is not None and time.monotonic() - self._last_seen > SECONDS_BEFORE_RESET:
            self.reset()

    def process(self, frame) -> FrameResult:
        faces = self.engine.detect(frame)
        if len(faces) != 1:
            self._face_missing()
            return FrameResult("none" if not faces else "multiple")

        face = faces[0]
        match = self.matcher.best_match(self.engine.embed(frame, face))
        if match is None:
            # A strong head turn can briefly lower the match score, so only
            # restart liveness if the face stays unrecognised for a while.
            self._face_missing()
            return FrameResult("unknown", face)

        self._last_seen = time.monotonic()
        if match[0] != self._current_pk:   # a different person: start over
            self.liveness.reset()
            self._current_pk = match[0]

        self.liveness.update(frame, face)
        if not self.liveness.is_live():
            return FrameResult("liveness", face, match, self.liveness.prompt,
                               self.liveness.debug_text)
        return FrameResult("accepted", face, match, debug=self.liveness.debug_text)
