"""Webcam enrolment and live recognition."""
import time
from pathlib import Path

import cv2

from . import attendance, db
from .config import settings
from .face_engine import FaceEngine, Matcher, to_blob
from .liveness import LivenessChecker

GREEN, RED, WHITE = (0, 200, 0), (0, 0, 220), (255, 255, 255)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _open_camera():
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {settings.camera_index}. Check CAMERA_INDEX in .env.")
    return cap


def _label(frame, text, box, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def enroll_from_camera(conn, engine: FaceEngine, student, samples: int = 25) -> int:
    """Capture `samples` embeddings, only from frames with exactly one face."""
    cap = _open_camera()
    vectors, last = [], 0.0
    print("Look at the camera and move your head slightly. Press q to cancel.")
    try:
        while len(vectors) < samples:
            ok, frame = cap.read()
            if not ok:
                break
            faces = engine.detect(frame)
            view = frame.copy()
            if len(faces) == 1:
                _label(view, f"{len(vectors)}/{samples}", faces[0].box, GREEN)
                if time.time() - last > 0.25:  # spread samples over ~6 seconds
                    vectors.append(engine.embed(frame, faces[0]))
                    last = time.time()
            elif len(faces) > 1:
                cv2.putText(view, "Only one person please", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            cv2.imshow("Enrolment", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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


def run_recognition(conn, engine: FaceEngine) -> None:
    matcher = Matcher(db.all_embeddings(conn))  # loaded once
    if not matcher.labels:
        print("No students enrolled yet.")
        return
    liveness = LivenessChecker()
    cap = _open_camera()
    print("Recognition running. Press q to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for face in engine.detect(frame):
                match = matcher.best_match(engine.embed(frame, face))
                liveness.update(frame, face)
                if match is None:
                    _label(frame, "Unknown", face.box, RED)
                    continue
                pk, sid, name, sim = match
                if not liveness.is_live():
                    _label(frame, f"{sid}: spoof?", face.box, RED)
                    continue
                if attendance.mark(conn, pk, sim):
                    print(f"  marked present: {sid} {name} ({sim:.2f})")
                _label(frame, f"{sid} {sim:.2f}", face.box, GREEN)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
