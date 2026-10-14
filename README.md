# EduVision Secure: Face-Recognition Attendance

A face-recognition attendance system rebuilt from scratch with security as the focus.
It is a rebuild and extension of my BS Cyber Security final year project
(*EduVision Attendance Tracker*, Islamia University of Bahawalpur, 2025).

> **Status:** Week 1 of 3. Core system done. Liveness detection (anti-spoofing) is next.

## Why rebuild it

A security review of the original project found these problems. This version fixes them:

| Original problem | Fix in this version |
|---|---|
| Hard-coded `admin/admin` login | No default account; first admin is created at setup |
| Passwords stored in plain text | bcrypt hashing (cost 12) |
| Unlimited login attempts | Account locks for 15 minutes after 5 failures |
| Database password and file paths hard-coded | Settings in `.env` (never committed) |
| New database connection per face per frame | Embeddings loaded once into memory |
| Attendance could be marked repeatedly | `UNIQUE(student, date)` constraint in the database |
| SQL built from strings | Parameterised queries only |
| No liveness check: a printed photo fools it | *Week 2:* blink-based liveness detection |
| Face data stored unencrypted | *Week 2:* encrypted embeddings |

## How it works

1. **Detect** faces with YuNet (OpenCV's built-in face detector).
2. **Embed** each face as a 128-number vector with SFace.
3. **Match** using cosine similarity against enrolled students (threshold 0.363).
4. **Mark** attendance once per student per day in SQLite.

```
eduvision/
  config.py       settings from .env
  db.py           SQLite schema and queries
  auth.py         admin accounts, bcrypt, lockout
  face_engine.py  detection, embeddings, matching
  camera.py       webcam enrolment and live recognition
  attendance.py   marking and CSV export
  liveness.py     anti-spoofing (Week 2)
  cli.py          commands
tests/            unit tests (pytest)
```

## Setup

Requires Python 3.10+ and a webcam.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_models.py # downloads and checksum-verifies the models
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

## Usage

```bash
python main.py create-admin                                   # first-time setup
python main.py add-student --id S21BINSE1M02014 --name "Asim Sohail"
python main.py enroll --id S21BINSE1M02014                    # webcam, 25 samples
python main.py run                                            # live attendance, q to quit
python main.py report --csv                                   # today's list + CSV
```

## Tests

```bash
python -m pytest
```

Tests cover password hashing, weak-password rejection, lockout, once-per-day
attendance, SQL-injection safety, and the match threshold.

## Roadmap

- [x] **Week 1:** database, secure login, enrolment, recognition, attendance
- [ ] **Week 2:** spoofing tests (photo, phone screen), blink liveness, encrypted embeddings, audit log
- [ ] **Week 3:** before/after false accept and false reject rates, dashboard, demo video

## Security results

*To be filled after Week 3 testing.*

| Attack | Accepted before liveness | Accepted after liveness |
|---|---|---|
| Printed photo | – | – |
| Photo on phone screen | – | – |

## Author

Asim Sohail, BS Cyber Security (IUB, 2021–2025)
