"""Attendance marking (once per student per day) and CSV export."""
import csv
import sqlite3
from datetime import datetime
from pathlib import Path


def mark(conn, student_pk: int, similarity: float, when: datetime | None = None) -> bool:
    """Mark a student present. Returns False if already marked today."""
    when = when or datetime.now()
    try:
        conn.execute(
            "INSERT INTO attendance (student_pk, date, time, similarity) VALUES (?, ?, ?, ?)",
            (student_pk, when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S"), similarity),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:  # UNIQUE(student_pk, date)
        return False


def report(conn, date: str):
    return conn.execute(
        "SELECT s.student_id, s.name, a.time, a.similarity "
        "FROM attendance a JOIN students s ON s.id = a.student_pk "
        "WHERE a.date = ? ORDER BY a.time",
        (date,),
    ).fetchall()


def export_csv(conn, date: str, out_path: Path) -> int:
    rows = report(conn, date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "student_id", "name", "time", "similarity"])
        for r in rows:
            w.writerow([date, r["student_id"], r["name"], r["time"], f"{r['similarity']:.3f}"])
    return len(rows)
