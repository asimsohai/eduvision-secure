"""Webcam enrolment and live recognition."""
import time
from pathlib import Path

import cv2

from . import attendance, db
from .config import settings
from .face_engine import FaceEngine, Matcher, to_blob
from .pipeline import Pipeline

GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 220), (255, 255, 255), (0, 210, 255)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _open_camera():
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {settings.camera_index}. Check CAMERA_INDEX in .env.")
    return cap


def read_frame(cap, attempts: int = 50):
    """Read one frame. Windows webcams often return a few empty frames while
    starting up, so retry for up to ~2.5 seconds before giving up."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError("The camera is not sending images. Close other apps using it and try again.")


def _should_stop(window: str) -> bool:
    """Stop on q, Q or Esc, or when the window's X button is clicked."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def mirror(frame):
    """Show the camera like a mirror, so turning to your left moves you left on screen."""
    return cv2.flip(frame, 1)


def _mirror_box(box, width):
    x, y, w, h = box
    return (width - x - w, y, w, h)


def _label(frame, text, box, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def enroll_from_camera(conn, engine: FaceEngine, student, samples: int = 25) -> int:
    """Capture `samples` embeddings, only from frames with exactly one face."""
    cap = _open_camera()
    vectors, last = [], 0.0
    print("Look at the camera and move your head slightly. Press q or Esc (or close the window) to cancel.")
    try:
        while len(vectors) < samples:
            frame = read_frame(cap)
            faces = engine.detect(frame)
            view = mirror(frame)
            if len(faces) == 1:
                _label(view, f"{len(vectors)}/{samples}",
                       _mirror_box(faces[0].box, frame.shape[1]), GREEN)
                if time.time() - last > 0.25:  # spread samples over ~6 seconds
                    vectors.append(engine.embed(frame, faces[0]))
                    last = time.time()
            elif len(faces) > 1:
                cv2.putText(view, "Only one person please", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            cv2.imshow("Enrolment", view)
            if _should_stop("Enrolment"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if vectors:
        db.add_embeddings(conn, student["id"], [to_blob(v) for v in vectors])
    return len(vectors)


def enroll_from_folder(conn, engine: FaceEngine, student, folder: Path) -> int:
    """Enrol from saved photos (one person per photo)."""
    blobs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(str(path))
        faces = engine.detect(image) if image is not None else []
        if len(faces) != 1:
            print(f"  skipped {path.name}: found {len(faces)} faces")
            continue
        blobs.append(to_blob(engine.embed(image, faces[0])))
    if blobs:
        db.add_embeddings(conn, student["id"], blobs)
    return len(blobs)


def _banner(frame, text, color=WHITE):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _footer(frame, text):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def _arrow(frame, direction: str) -> None:
    """Big arrow on the side the person should turn towards (mirrored view)."""
    h, w = frame.shape[:2]
    y = h // 2
    if direction == "left":
        start, end = (int(w * 0.3), y), (int(w * 0.08), y)
    else:
        start, end = (int(w * 0.7), y), (int(w * 0.92), y)
    cv2.arrowedLine(frame, start, end, YELLOW, 8, tipLength=0.35)


def draw_result(frame, result):
    """Return a mirrored copy of the frame with boxes, prompts and arrows drawn."""
    view = mirror(frame)
    width = frame.shape[1]
    box = _mirror_box(result.face.box, width) if result.face is not None else None
    if result.debug:
        _footer(view, result.debug)
    if result.status == "none":
        _banner(view, "Look at the camera")
    elif result.status == "multiple":
        _banner(view, "One person at a time please", RED)
    elif result.status == "unknown":
        _label(view, "Unknown", box, RED)
    elif result.status == "liveness":
        _, sid, _, sim = result.match
        _label(view, f"{sid} {sim:.2f}", box, YELLOW)
        alert = result.prompt.startswith(("Wrong", "Too slow"))
        _banner(view, result.prompt, RED if alert else YELLOW)
        if "your LEFT" in result.prompt:
            _arrow(view, "left")
        elif "your RIGHT" in result.prompt:
            _arrow(view, "right")
    elif result.status == "accepted":
        _, sid, _, sim = result.match
        _label(view, f"{sid} {sim:.2f} LIVE", box, GREEN)
    return view


def run_recognition(conn, engine: FaceEngine, mode: str | None = None) -> None:
    mode = mode or settings.liveness_mode
    matcher = Matcher(db.all_embeddings(conn))  # loaded once
    if not matcher.labels:
        print("No students enrolled yet.")
        return
    pipeline = Pipeline(engine, matcher, mode)
    cap = _open_camera()
    print(f"Recognition running (liveness: {mode}). "
          "Press q or Esc (or close the window) to stop.")
    try:
        while True:
            frame = read_frame(cap)
            result = pipeline.process(frame)
            if result.status == "accepted":
                pk, sid, name, sim = result.match
                if attendance.mark(conn, pk, sim):
                    print(f"  marked present: {sid} {name} ({sim:.2f})")
                pipeline.reset()  # the next person must prove liveness again
            cv2.imshow("Attendance", draw_result(frame, result))
            if _should_stop("Attendance"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def pose_test(engine: FaceEngine) -> None:
    """Show the live head-turn measurement, to check the challenge works on this camera."""
    from .liveness import head_turn_offset, classify_pose

    cap = _open_camera()
    baseline = None
    print("Look straight, then turn your head to YOUR left and right. "
          "Press c to re-centre, q or Esc to quit.")
    try:
        while True:
            raw = read_frame(cap)
            faces = engine.detect(raw)
            frame = mirror(raw)
            if len(faces) == 1:
                off = head_turn_offset(faces[0])
                if baseline is None:
                    baseline = off
                rel = off - baseline
                pose = classify_pose(rel, settings.turn_threshold)
                color = GREEN if pose in ("left", "right") else WHITE
                _label(frame, pose.upper(), _mirror_box(faces[0].box, raw.shape[1]), color)
                _banner(frame, f"turn: {rel:+.2f}   detected: {pose.upper()}", color)
                _footer(frame, f"needs +/-{settings.turn_threshold:.2f}. "
                               "Turn to YOUR left: should say LEFT. c = re-centre")
            else:
                _banner(frame, "Show exactly one face")
            cv2.imshow("Pose test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                baseline = None
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("Pose test", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
