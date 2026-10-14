"""Command-line interface. Every command except setup needs an admin login."""
import argparse
import getpass
import sqlite3
import sys
from datetime import date
from pathlib import Path

from . import attendance, auth, db
from .config import PROJECT_ROOT, settings


def _require_login(conn) -> None:
    if auth.admin_count(conn) == 0:
        sys.exit("No admin exists yet. Run: python main.py create-admin")
    username = input("Admin username: ")
    password = getpass.getpass("Password: ")
    try:
        ok = auth.login(conn, username, password)
    except auth.AuthError as e:
        sys.exit(str(e))
    if not ok:
        sys.exit("Invalid username or password.")


def cmd_create_admin(conn, args):
    if auth.admin_count(conn) > 0:
        print("An admin already exists; log in to add another.")
        _require_login(conn)
    username = input("New admin username: ")
    password = getpass.getpass("New password: ")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    try:
        auth.create_admin(conn, username, password)
    except (auth.AuthError, sqlite3.IntegrityError) as e:
        sys.exit(f"Could not create admin: {e}")
    print(f"Admin '{username}' created.")


def cmd_add_student(conn, args):
    try:
        db.add_student(conn, args.id, args.name)
    except sqlite3.IntegrityError:
        sys.exit(f"Student {args.id} already exists.")
    print(f"Added {args.id} {args.name}. Next: python main.py enroll --id {args.id}")


def cmd_students(conn, args):
    for s in db.list_students(conn):
        print(f"{s['student_id']:<20} {s['name']:<30} {s['samples']} face samples")


def cmd_enroll(conn, args):
    from .camera import enroll_from_camera, enroll_from_folder
    from .face_engine import FaceEngine

    student = db.get_student(conn, args.id)
    if student is None:
        sys.exit(f"Unknown student {args.id}. Add them first with add-student.")
    engine = FaceEngine()
    if args.images:
        n = enroll_from_folder(conn, engine, student, Path(args.images))
    else:
        n = enroll_from_camera(conn, engine, student, args.samples)
    print(f"Stored {n} face samples for {args.id}.")


def cmd_run(conn, args):
    from .camera import run_recognition
    from .face_engine import FaceEngine

    run_recognition(conn, FaceEngine())


def cmd_report(conn, args):
    day = args.date or date.today().isoformat()
    rows = attendance.report(conn, day)
    print(f"Attendance for {day}: {len(rows)} present")
    for r in rows:
        print(f"  {r['time']}  {r['student_id']:<20} {r['name']}")
    if args.csv:
        out = Path(args.csv) if args.csv != "auto" else PROJECT_ROOT / "data" / f"attendance_{day}.csv"
        attendance.export_csv(conn, day, out)
        print(f"Saved {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Secure face-recognition attendance")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("create-admin", help="create an admin account")

    s = sub.add_parser("add-student", help="register a student")
    s.add_argument("--id", required=True, help="roll number")
    s.add_argument("--name", required=True)

    sub.add_parser("students", help="list students")

    s = sub.add_parser("enroll", help="capture a student's face")
    s.add_argument("--id", required=True)
    s.add_argument("--samples", type=int, default=25)
    s.add_argument("--images", help="enrol from a folder of photos instead of the webcam")

    sub.add_parser("run", help="start live recognition")

    s = sub.add_parser("report", help="show attendance for a day")
    s.add_argument("--date", help="YYYY-MM-DD (default today)")
    s.add_argument("--csv", nargs="?", const="auto", help="also save as CSV")
    return p


COMMANDS = {
    "create-admin": (cmd_create_admin, False),
    "add-student": (cmd_add_student, True),
    "students": (cmd_students, True),
    "enroll": (cmd_enroll, True),
    "run": (cmd_run, True),
    "report": (cmd_report, True),
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    conn = db.connect(settings.db_path)
    db.init_db(conn)
    func, needs_login = COMMANDS[args.command]
    if needs_login:
        _require_login(conn)
    func(conn, args)
