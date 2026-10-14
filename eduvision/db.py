"""SQLite storage. All queries are parameterised (no string-built SQL)."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS admins (
    id              INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL UNIQUE,
    password_hash   BLOB    NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY,
    student_id  TEXT NOT NULL UNIQUE,   -- roll number, e.g. S21BINSE1M02014
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY,
    student_pk  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL,       -- 128 float32 values
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY,
    student_pk  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,       -- YYYY-MM-DD
    time        TEXT    NOT NULL,       -- HH:MM:SS
    similarity  REAL    NOT NULL,
    UNIQUE (student_pk, date)           -- at most one mark per student per day
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---- students -------------------------------------------------------------

def add_student(conn, student_id: str, name: str) -> int:
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO students (student_id, name) VALUES (?, ?)",
            (student_id.strip(), name.strip()),
        )
    return cur.lastrowid


def get_student(conn, student_id: str):
    return conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()


def list_students(conn):
    return conn.execute(
        "SELECT s.*, COUNT(f.id) AS samples FROM students s "
        "LEFT JOIN face_embeddings f ON f.student_pk = s.id "
        "GROUP BY s.id ORDER BY s.student_id"
    ).fetchall()


# ---- face embeddings ------------------------------------------------------

def add_embeddings(conn, student_pk: int, blobs: list[bytes]) -> None:
    with transaction(conn):
        conn.executemany(
            "INSERT INTO face_embeddings (student_pk, embedding) VALUES (?, ?)",
            [(student_pk, b) for b in blobs],
        )


def all_embeddings(conn):
    return conn.execute(
        "SELECT f.student_pk, f.embedding, s.student_id, s.name "
        "FROM face_embeddings f JOIN students s ON s.id = f.student_pk"
    ).fetchall()
