from datetime import datetime, timedelta

import numpy as np
import pytest

from eduvision import attendance, auth, db
from eduvision.face_engine import Matcher, to_blob


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.init_db(c)
    return c


GOOD_PW = "Correct-Horse-9"


def test_password_is_hashed_not_plain(conn):
    auth.create_admin(conn, "teacher", GOOD_PW)
    stored = conn.execute("SELECT password_hash FROM admins").fetchone()[0]
    assert GOOD_PW.encode() not in stored
    assert stored.startswith(b"$2b$")


def test_weak_password_rejected(conn):
    with pytest.raises(auth.AuthError):
        auth.create_admin(conn, "teacher", "admin")


def test_login(conn):
    auth.create_admin(conn, "teacher", GOOD_PW)
    assert auth.login(conn, "teacher", GOOD_PW)
    assert not auth.login(conn, "teacher", "wrong-password")
    assert not auth.login(conn, "nobody", GOOD_PW)


def test_lockout_after_five_failures(conn):
    auth.create_admin(conn, "teacher", GOOD_PW)
    now = datetime(2026, 10, 15, 9, 0)
    for _ in range(5):
        assert not auth.login(conn, "teacher", "wrong-password", now=now)
    with pytest.raises(auth.AuthError):  # locked, even with the right password
        auth.login(conn, "teacher", GOOD_PW, now=now + timedelta(minutes=1))
    assert auth.login(conn, "teacher", GOOD_PW, now=now + timedelta(minutes=16))


def test_attendance_once_per_day(conn):
    pk = db.add_student(conn, "S001", "Test Student")
    day1 = datetime(2026, 10, 15, 9, 0)
    assert attendance.mark(conn, pk, 0.8, day1)
    assert not attendance.mark(conn, pk, 0.9, day1.replace(hour=11))
    assert attendance.mark(conn, pk, 0.8, day1 + timedelta(days=1))
    assert len(attendance.report(conn, "2026-10-15")) == 1


def test_sql_injection_in_student_id_is_harmless(conn):
    db.add_student(conn, "x'); DROP TABLE students;--", "Bobby")
    assert len(db.list_students(conn)) == 1


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_matcher_threshold(conn):
    rng = np.random.default_rng(0)
    alice = _unit(rng.normal(size=128))
    pk = db.add_student(conn, "S001", "Alice")
    db.add_embeddings(conn, pk, [to_blob(alice)])
    m = Matcher(db.all_embeddings(conn))
    close = _unit(alice + 0.1 * rng.normal(size=128))
    stranger = _unit(rng.normal(size=128))
    assert m.best_match(close)[1] == "S001"
    assert m.best_match(stranger) is None
