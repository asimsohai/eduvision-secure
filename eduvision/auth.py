"""Admin authentication.

Fixes from the original project:
- no hard-coded admin/admin account; the first admin is created by the user
- passwords stored as bcrypt hashes, never plain text
- account locks for a while after repeated failed logins
- same response and similar timing for unknown users and wrong passwords
"""
from datetime import datetime, timedelta

import bcrypt

from .config import settings
from .db import transaction

MIN_PASSWORD_LENGTH = 10
# Hash used when the username does not exist, so the check takes the same time.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt())


class AuthError(Exception):
    pass


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.lower() == password or password.isalpha():
        raise AuthError("Password needs a mix of upper/lower case and a digit or symbol.")


def admin_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0]


def create_admin(conn, username: str, password: str) -> None:
    username = username.strip()
    if not username:
        raise AuthError("Username cannot be empty.")
    _validate_password(password)
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
    with transaction(conn):
        conn.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            (username, pw_hash),
        )


def login(conn, username: str, password: str, now: datetime | None = None) -> bool:
    """Return True on success. Raises AuthError if the account is locked."""
    now = now or datetime.now()
    row = conn.execute(
        "SELECT * FROM admins WHERE username = ?", (username.strip(),)
    ).fetchone()

    if row is None:
        bcrypt.checkpw(password.encode(), _DUMMY_HASH)  # equal timing
        return False

    if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now:
        raise AuthError("Account locked after too many failed attempts. Try later.")

    if bcrypt.checkpw(password.encode(), row["password_hash"]):
        with transaction(conn):
            conn.execute(
                "UPDATE admins SET failed_attempts = 0, locked_until = NULL WHERE id = ?",
                (row["id"],),
            )
        return True

    attempts = row["failed_attempts"] + 1
    locked_until = None
    if attempts >= settings.max_failed_logins:
        locked_until = (now + timedelta(minutes=settings.lockout_minutes)).isoformat()
        attempts = 0
    with transaction(conn):
        conn.execute(
            "UPDATE admins SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            (attempts, locked_until, row["id"]),
        )
    return False
