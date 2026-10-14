"""Security evaluation: how often is the system fooled, with each liveness mode?

Liveness modes compared: off (no protection), blink, challenge (random head turns).

Each TRIAL gives the system a few seconds to accept what is in front of the
camera. A trial is "accepted" if any frame in that time would have marked
attendance. Attendance is NOT actually recorded during evaluation.

Labels:
  genuine  - the real enrolled person, live
  print    - a printed photo of the enrolled person
  screen   - a photo of the enrolled person shown on a phone/laptop screen
  video    - a video of the enrolled person (blinking) on a screen

Metrics (terms from ISO/IEC 30107-3, the presentation-attack standard):
  APCER - Attack Presentation Classification Error Rate:
          share of ATTACK trials that were wrongly accepted. Lower is better.
  BPCER - Bona fide Presentation Classification Error Rate:
          share of GENUINE trials that were wrongly rejected. Lower is better.
"""
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from . import db
from .camera import _banner, _open_camera, draw_result, mirror, read_frame, GREEN, YELLOW
from .face_engine import FaceEngine, Matcher
from .pipeline import Pipeline

LABELS = ("genuine", "print", "screen", "video")
FIELDS = ["timestamp", "label", "liveness", "trial", "accepted",
          "seconds_to_accept", "best_similarity", "student_id"]
WINDOW = "Evaluation"


def _wait_for_start(cap, text: str) -> bool:
    """Show the camera with instructions until SPACE (start) or q/Esc (quit)."""
    while True:
        frame = mirror(read_frame(cap))
        _banner(frame, text)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False


def run_trials(conn, engine: FaceEngine, label: str, trials: int, seconds: float,
               mode: str, out_csv: Path) -> list[dict]:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    matcher = Matcher(db.all_embeddings(conn))
    if not matcher.labels:
        raise RuntimeError("No students enrolled yet.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    rows = []
    cap = _open_camera()
    try:
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()

            for t in range(1, trials + 1):
                prompt = (f"[{label}] trial {t}/{trials} - liveness: {mode}"
                          " - SPACE to start, q to stop")
                if not _wait_for_start(cap, prompt):
                    print("Stopped early.")
                    break

                pipeline = Pipeline(engine, matcher, mode)
                start, best_sim, accepted_at, student = time.time(), 0.0, None, ""
                while time.time() - start < seconds:
                    frame = read_frame(cap)
                    result = pipeline.process(frame)
                    if result.match:
                        best_sim = max(best_sim, result.match[3])
                        student = result.match[1]
                    view = draw_result(frame, result)
                    left = seconds - (time.time() - start)
                    if result.status == "accepted":
                        accepted_at = time.time() - start
                        _banner(view, f"ACCEPTED after {accepted_at:.1f}s", GREEN)
                    elif result.status != "liveness":
                        _banner(view, f"Trial {t}: {left:.0f}s left", YELLOW)
                    else:
                        cv2.putText(view, f"{left:.0f}s", (view.shape[1] - 60, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
                    cv2.imshow(WINDOW, view)
                    cv2.waitKey(1)
                    if accepted_at is not None:
                        cv2.waitKey(700)  # let the result be seen
                        break

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "label": label,
                    "liveness": mode,
                    "trial": t,
                    "accepted": int(accepted_at is not None),
                    "seconds_to_accept": f"{accepted_at:.2f}" if accepted_at is not None else "",
                    "best_similarity": f"{best_sim:.3f}",
                    "student_id": student,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                detail = ""
                if hasattr(pipeline.liveness, "summary"):
                    detail = f" [{pipeline.liveness.summary()}]"
                lost = pipeline.missing_frames
                if lost:
                    detail += f" [face lost/unrecognised in {lost} frames]"
                print(f"  trial {t}: {'ACCEPTED' if row['accepted'] else 'rejected'}"
                      f" (best similarity {best_sim:.2f}){detail}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Group trials by (liveness, label) and compute acceptance rates."""
    groups = defaultdict(list)
    for r in rows:
        mode = "blink" if r["liveness"] == "on" else r["liveness"]  # early runs said "on"
        groups[(mode, r["label"])].append(r)

    summary = []
    for liveness in ("off", "blink", "challenge"):
        for label in LABELS:
            trials = groups.get((liveness, label))
            if not trials:
                continue
            accepted = sum(int(r["accepted"]) for r in trials)
            times = [float(r["seconds_to_accept"]) for r in trials if r["seconds_to_accept"]]
            n = len(trials)
            summary.append({
                "liveness": liveness,
                "label": label,
                "trials": n,
                "accepted": accepted,
                "acceptance_rate": accepted / n,
                # genuine: wrongly rejected = BPCER; attacks: wrongly accepted = APCER
                "metric": "BPCER" if label == "genuine" else "APCER",
                "error_rate": (n - accepted) / n if label == "genuine" else accepted / n,
                "mean_seconds": sum(times) / len(times) if times else None,
            })
    return summary


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(summary: list[dict]) -> str:
    lines = [
        "| Liveness | Presentation | Trials | Accepted | Error metric | Error rate | Mean time to accept |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        t = f"{s['mean_seconds']:.1f}s" if s["mean_seconds"] is not None else "-"
        lines.append(
            f"| {s['liveness'].upper()} | {s['label']} | {s['trials']} | {s['accepted']} "
            f"| {s['metric']} | {s['error_rate']:.0%} | {t} |"
        )
    return "\n".join(lines)


def clear_mode(path: Path, mode: str) -> int:
    """Delete all trials of one liveness mode (e.g. after changing that method)."""
    rows = read_results(path)
    keep = [r for r in rows if r["liveness"] != mode]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(keep)
    return len(rows) - len(keep)
