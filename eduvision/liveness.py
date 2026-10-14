"""Liveness detection (anti-spoofing): is a real, live person in front of the camera?

Two methods are implemented so they can be compared in the evaluation:

1. BLINK (BlinkLivenessChecker) - the person must blink. Weakness found in
   testing: on a phone screen the eye detector flickers, and flicker can look
   like a blink, so screen photos often passed.

2. CHALLENGE (ChallengeLivenessChecker, the default) - the person must turn
   their head in a RANDOM sequence, e.g. LEFT, RIGHT, LEFT. This uses 3D
   geometry: when a real head turns, the nose moves sideways relative to the
   eyes. A flat photo, even when tilted, keeps the nose in the same place
   relative to the eyes, so it cannot pass. Because the sequence is random,
   a pre-recorded video only matches it by luck (1 in 2^steps).

--- Blink method details ---

Idea: a printed photo or a still image on a phone screen cannot blink, but a
real person blinks every few seconds. So before attendance is marked, the
person must blink while the camera watches.

How a blink is detected, frame by frame:
1. Take the upper part of the face box, where the eyes are.
2. Run OpenCV's Haar eye detector there. It finds OPEN eyes; when the eyes
   close, it finds none.
3. A blink is: eyes open for a few frames -> closed for a short moment ->
   open again. Closed too briefly (detector flicker) or too long (eyes shut,
   or face turned away) does not count.

The decision logic (BlinkCounter) is separate from the eye detector so it can
be unit-tested without a camera.

Known limitation: a VIDEO of the person blinking (replay attack) can still
pass. Measuring that is part of the evaluation.
"""
import random
import time

import cv2

from .config import settings


class BlinkCounter:
    """State machine that turns a stream of eyes-open/closed values into blinks."""

    def __init__(self, min_closed: int = 2, max_closed: int = 8, min_open: int = 2):
        self.min_closed = min_closed  # frames; filters out detector flicker
        self.max_closed = max_closed  # frames; longer is not a blink
        self.min_open = min_open      # open frames needed before a closure counts
        self.reset()

    def reset(self) -> None:
        self.blinks = 0
        self._open_run = 0
        self._closed_run = 0

    def update(self, eyes_open: bool) -> bool:
        """Feed one frame. Returns True if this frame completed a blink."""
        if not eyes_open:
            self._closed_run += 1
            return False

        completed = (
            self._open_run >= self.min_open
            and self.min_closed <= self._closed_run <= self.max_closed
        )
        if completed:
            self.blinks += 1
            self._open_run = 1
        elif self._closed_run > 0:
            self._open_run = 1      # closure was invalid: start counting again
        else:
            self._open_run += 1
        self._closed_run = 0
        return completed


class BlinkLivenessChecker:
    enabled = True

    def __init__(self, required_blinks: int | None = None):
        self.required_blinks = required_blinks or settings.required_blinks
        self.counter = BlinkCounter(
            min_closed=settings.blink_min_closed_frames,
            max_closed=settings.blink_max_closed_frames,
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if self.eye_detector.empty():
            raise RuntimeError("Could not load OpenCV's eye detector (haarcascade_eye.xml).")
        self.last_eyes = []  # for drawing/debugging

    def reset(self) -> None:
        self.counter.reset()

    def eyes_open(self, frame, box) -> bool:
        x, y, w, h = box
        H, W = frame.shape[:2]
        top, bottom = max(0, y + int(h * 0.15)), min(H, y + int(h * 0.55))
        left, right = max(0, x), min(W, x + w)
        if bottom <= top or right <= left:
            return False
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        size = max(10, w // 8)
        eyes = self.eye_detector.detectMultiScale(gray, 1.1, 5, minSize=(size, size))
        self.last_eyes = [(left + ex, top + ey, ew, eh) for ex, ey, ew, eh in eyes]
        return len(eyes) >= 1

    def update(self, frame, face) -> None:
        self.counter.update(self.eyes_open(frame, face.box))

    def is_live(self) -> bool:
        return self.counter.blinks >= self.required_blinks

    @property
    def prompt(self) -> str:
        return "Please blink to confirm you are live"

    @property
    def debug_text(self) -> str:
        return f"blinks: {self.counter.blinks}"


# --- Challenge method -------------------------------------------------------

def head_turn_offset(face) -> float:
    """Sideways position of the nose relative to the eyes, from YuNet landmarks.

    0 = looking straight at the camera. Positive = head turned to the person's
    own LEFT, negative = to their RIGHT (camera image is not mirrored).
    Scaled by the distance between the eyes, so it does not depend on how far
    the person is from the camera.
    """
    r = face.raw
    eye1_x, eye2_x, nose_x = float(r[4]), float(r[6]), float(r[8])
    eye_mid = (eye1_x + eye2_x) / 2
    eye_dist = abs(eye2_x - eye1_x)
    if eye_dist < 1:
        return 0.0
    offset = (nose_x - eye_mid) / eye_dist
    return -offset if settings.challenge_flip else offset


CENTER_BAND = 0.6  # |offset| below threshold * CENTER_BAND counts as straight


def classify_pose(offset: float, threshold: float) -> str:
    if offset > threshold:
        return "left"
    if offset < -threshold:
        return "right"
    if abs(offset) < threshold * CENTER_BAND:
        return "center"
    return "between"


class ChallengeLivenessChecker:
    """Random head-turn challenge.

    Sequence for steps = [left, right, left]:
      look straight -> turn LEFT -> back to centre -> turn RIGHT -> centre
      -> turn LEFT -> live.
    Calibration: a webcam is rarely exactly in front of the face (laptops sit
    below it), so "straight" is measured from the person's own first steady
    frames (the baseline), and turns are measured relative to that.
    Each pose must be held for `hold` frames. Turning the wrong way, or taking
    longer than `step_seconds` for a step, fails the attempt.

    Attempt limit: after `max_attempts` failed attempts the checker locks for
    `lockout_seconds`. Without this limit (the first version) an attacker could
    retry endlessly, and a looping video or a wobbling photo eventually matched
    the random order by chance - the evaluation measured 100% attack success.
    """

    enabled = True

    def __init__(self, steps=None, hold=None, step_seconds=None, threshold=None,
                 rng=None, clock=time.monotonic, max_attempts=None, lockout_seconds=None):
        self.n_steps = steps or settings.challenge_steps
        self.hold = hold or settings.challenge_hold_frames
        self.step_seconds = step_seconds or settings.challenge_step_seconds
        self.threshold = threshold or settings.turn_threshold
        self.max_attempts = max_attempts or settings.challenge_max_attempts
        self.lockout_seconds = lockout_seconds or settings.challenge_lockout_seconds
        self.locked_until = None
        self.rng = rng or random.SystemRandom()  # unpredictable to an attacker
        self.clock = clock
        self.last_eyes: list = []
        self.failures = 0
        self.fail_reasons = {"timeout": 0, "wrong_turn": 0}
        self._last_fail = None
        self.best_progress = 0
        self.baseline = None
        self.last_offset = 0.0
        self.last_pose = "-"
        self._calib: list[float] = []
        self.reset()

    def reset(self, keep_baseline: bool = False) -> None:
        if not keep_baseline:
            self.baseline = None
            self._calib = []
        self.steps = [self.rng.choice(["left", "right"]) for _ in range(self.n_steps)]
        self.index = 0
        self.need_center = True   # every turn starts from looking straight
        self._run_pose, self._run = None, 0
        self._step_started = self.clock()
        self.live = False

    def _fail(self, reason: str) -> None:
        self.failures += 1
        self.fail_reasons[reason] += 1
        self._last_fail = (reason, self.clock())
        if self.failures >= self.max_attempts:
            # too many failed attempts: lock, like the login lockout
            self.locked_until = self.clock() + self.lockout_seconds
            self.failures = 0
        self.reset(keep_baseline=True)   # same person: keep their calibration

    @property
    def locked(self) -> bool:
        return self.locked_until is not None and self.clock() < self.locked_until

    def update_pose(self, pose: str) -> None:
        """Feed one frame's pose (center/left/right/between)."""
        if self.live or self.locked:
            return
        if self.locked_until is not None:          # lockout just ended: fresh start
            self.locked_until = None
            self.reset(keep_baseline=True)
        if self.clock() - self._step_started > self.step_seconds:
            self._fail("timeout")
            return

        self._run = self._run + 1 if pose == self._run_pose else 1
        self._run_pose = pose
        if self._run < self.hold:
            return

        if self.need_center:
            if pose == "center":
                self.need_center = False
                self._step_started = self.clock()
            return

        expected = self.steps[self.index]
        if pose == expected:
            self.index += 1
            self.best_progress = max(self.best_progress, self.index)
            self.need_center = True
            self._step_started = self.clock()
            self._run = 0
            if self.index == len(self.steps):
                self.live = True
        elif pose in ("left", "right"):
            self._fail("wrong_turn")  # turned the wrong way: new random challenge

    def update_offset(self, offset: float) -> None:
        """Feed one frame's raw offset; calibrates first, then classifies."""
        self.last_offset = offset
        if self.baseline is None:
            # wait for `hold` steady frames to learn this person's straight pose
            self._calib = (self._calib + [offset])[-self.hold:]
            steady = (len(self._calib) == self.hold
                      and max(self._calib) - min(self._calib) < self.threshold * CENTER_BAND)
            if steady:
                self.baseline = sum(self._calib) / len(self._calib)
                self._step_started = self.clock()
            self.last_pose = "calibrating"
            return
        pose = classify_pose(offset - self.baseline, self.threshold)
        self.last_pose = pose
        self.update_pose(pose)

    def update(self, frame, face) -> None:
        self.update_offset(head_turn_offset(face))

    def is_live(self) -> bool:
        return self.live

    def summary(self) -> str:
        """One line explaining what happened, printed after each evaluation trial."""
        calib = "calibrated" if self.baseline is not None else "restarted/not calibrated at end"
        lock = "; LOCKED (attempt limit reached)" if self.locked_until is not None else ""
        return (f"{calib}; best progress {self.best_progress}/{self.n_steps}; "
                f"timeouts {self.fail_reasons['timeout']}, wrong turns {self.fail_reasons['wrong_turn']}{lock}")

    @property
    def debug_text(self) -> str:
        rel = self.last_offset - (self.baseline or 0.0)
        return f"pose: {self.last_pose}  turn: {rel:+.2f}  (needs +/-{self.threshold:.2f})"

    @property
    def prompt(self) -> str:
        if self.live:
            return "Verified"
        done = f"({self.index}/{len(self.steps)})"
        if self.locked:
            wait = self.locked_until - self.clock()
            return f"Liveness check failed. Try again in {wait:.0f}s"
        if self._last_fail and self.clock() - self._last_fail[1] < 2.0:
            why = "Wrong way!" if self._last_fail[0] == "wrong_turn" else "Too slow!"
            return f"{why} New challenge - look straight"
        if self.baseline is None:
            return "Look straight at the camera and keep still"
        if self.need_center:
            return f"Look straight at the camera {done}"
        return f"Turn your head to your {self.steps[self.index].upper()} {done}"


class NoLivenessChecker:
    """Used to measure the system WITHOUT protection (the 'before' results)."""

    enabled = False
    last_eyes: list = []

    def reset(self) -> None:
        pass

    def update(self, frame, face) -> None:
        pass

    def is_live(self) -> bool:
        return True

    prompt = ""
    debug_text = ""


MODES = ("off", "blink", "challenge")


def make_checker(mode: str):
    if mode == "off":
        return NoLivenessChecker()
    if mode == "blink":
        return BlinkLivenessChecker()
    if mode == "challenge":
        return ChallengeLivenessChecker()
    raise ValueError(f"Unknown liveness mode '{mode}'. Use one of {MODES}.")
