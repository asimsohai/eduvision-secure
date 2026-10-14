# EduVision Secure: Face-Recognition Attendance

A face-recognition attendance system rebuilt from scratch with security as the focus.
It is a rebuild and extension of my BS Cyber Security final year project
(*EduVision Attendance Tracker*, Islamia University of Bahawalpur, 2025).

> **Status:** Week 2 of 3. Core system, blink check, hardened head-turn challenge and evaluation tool done. Formal attack evaluation pending.

## Why rebuild it

The original EduVision code (2025) is not published here. A security review of it found the problems below, and this version was written from scratch to fix them:

| Original problem | Fix in this version |
|---|---|
| Hard-coded `admin/admin` login | No default account; first admin is created at setup |
| Passwords stored in plain text | bcrypt hashing (cost 12) |
| Unlimited login attempts | Account locks for 15 minutes after 5 failures |
| Database password and file paths hard-coded | Settings in `.env` (never committed) |
| New database connection per face per frame | Embeddings loaded once into memory |
| Attendance could be marked repeatedly | `UNIQUE(student, date)` constraint in the database |
| SQL built from strings | Parameterised queries only |
| No liveness check: a photo fools it | Liveness detection: blink check and a random head-turn challenge |
| Face data stored unencrypted | *Week 2:* encrypted embeddings |

## How it works

1. **Detect** faces with YuNet (OpenCV's built-in face detector).
2. **Embed** each face as a 128-number vector with SFace.
3. **Match** using cosine similarity against enrolled students (threshold 0.363).
4. **Check liveness** with a random head-turn challenge, e.g. *LEFT, RIGHT, LEFT*.
   When a real head turns, the nose moves sideways relative to the eyes (3D
   geometry, measured from YuNet's facial landmarks). A flat photo, even when
   tilted, keeps the nose in place relative to the eyes, so it cannot follow the
   challenge. The order is random, so a pre-recorded video matches only by luck
   (1 in 8 for 3 steps). A simpler blink check is also implemented for comparison.
5. **Mark** attendance once per student per day in SQLite.

```
eduvision/
  config.py       settings from .env
  db.py           SQLite schema and queries
  auth.py         admin accounts, bcrypt, lockout
  face_engine.py  detection, embeddings, matching
  camera.py       webcam enrolment and live recognition
  attendance.py   marking and CSV export
  liveness.py     anti-spoofing: blink check and head-turn challenge
  pipeline.py     per-frame decision: detect, match, liveness
  evaluation.py   attack trials and APCER/BPCER metrics
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
python main.py run                                            # live attendance (follow the head-turn prompts), q to quit
python main.py report --csv                                   # today's list + CSV
```

## Security evaluation

Each trial gives the system a fixed time (8 s, or 15 s for the challenge) to
accept what is in front of the camera; attendance is not recorded during
evaluation. Attacks use images of the enrolled person.

```bash
python main.py evaluate --label genuine --liveness off       # no protection
python main.py evaluate --label screen  --liveness off       # photo on a phone screen
python main.py evaluate --label genuine --liveness blink
python main.py evaluate --label screen  --liveness blink
python main.py evaluate --label video   --liveness blink     # video of the person blinking
python main.py evaluate --label genuine --liveness challenge
python main.py evaluate --label screen  --liveness challenge
python main.py evaluate --label video   --liveness challenge # video of the person turning their head
python main.py eval-report                                   # results table
python main.py eval-clear --liveness challenge               # delete one mode's trials after changing it
```

Metrics follow ISO/IEC 30107-3 (biometric presentation attack detection):

- **APCER**: share of attack attempts wrongly accepted (lower is better)
- **BPCER**: share of genuine attempts wrongly rejected (lower is better)

## Tests

```bash
python -m pytest
```

Tests cover password hashing, weak-password rejection, lockout, once-per-day
attendance, SQL-injection safety, the match threshold, the blink state machine,
the head-turn challenge (correct sequence, static photo, wrong direction,
timeout, calibration, attempt limit and lockout, brute-force simulation) and the
evaluation metrics.

## Roadmap

- [x] **Week 1:** database, secure login, enrolment, recognition, attendance
- [x] **Week 2:** blink liveness, attack evaluation tool (APCER/BPCER), random head-turn challenge
- [ ] Formal attack evaluation (photo, screen, video) of all liveness modes
- [ ] **Week 2:** encrypted face embeddings, audit log
- [ ] **Week 3:** before/after false accept and false reject rates, dashboard, demo video

## Security results

A formal attack evaluation (10 trials per mode and attack, using the
`evaluate` command above) is still to be run and will be reported here. Until
then, the defences are supported by unit tests, analysis and simulation, as
described below.

### Design reasoning: why a head-turn challenge

- Without liveness detection, any recognisable photo of an enrolled student is
  accepted, because face recognition alone cannot tell a live face from an image.
- A blink check is the simplest defence, but it depends on an eye detector; on a
  phone screen such detectors can flicker between "eyes found" and "no eyes",
  and flicker can look like a blink. The blink check is kept as a baseline for
  comparison in the evaluation.
- The head-turn challenge relies on 3D geometry instead: a real head turning
  moves the nose sideways relative to the eyes, while a flat image does not.

### Hardening the head-turn challenge (analysis and simulation)

- **Attempt limit:** the first version generated a new random challenge after
  every failure, with no limit. In simulation, a video alternating LEFT/RIGHT
  passed 99.6% of 15-second sessions, because it eventually matched the random
  order by chance. As with the login lockout, the challenge now allows 2
  attempts, then locks for 30 seconds; the same simulation falls to 23%.
- **Rotation:** measuring only horizontal positions meant that rotating a photo
  like a steering wheel moved the nose sideways in the image. On a test image a
  20 degree rotation read +0.154, above the 0.12 threshold: a fake head turn. The
  nose is now measured along the line between the eyes, frames with the face
  rotated more than 15 degrees are ignored, and the median of 3 frames reduces
  landmark jitter. In a simulation of a photo being rotated and tilted for 15
  seconds, sessions passing fell from 13/300 to 5/300.

Making the challenge usable for real people took further fixes, found from the
per-trial diagnostics that `evaluate` prints during development runs:

- **Calibration:** a laptop webcam sits below the face, so "looking straight" is
  learned from the person's first steady frames instead of assumed to be zero.
- **Threshold:** in the pose test, turning left reached only 0.15 on this camera,
  so the turn threshold was lowered from 0.18 to 0.12. Tilted flat test images
  measured below 0.08.
- **Mirror view and arrows:** the unmirrored camera view caused left/right mix-ups.
- **Face loss when turned:** the face detector threshold was lowered (0.9 to 0.75)
  and short detection gaps no longer restart the challenge.

## Limitations

- **Blink check:** vulnerable in principle to eye-detector flicker on screens and
  to videos of the person blinking; kept as a baseline for comparison.
- **Landmark noise:** YuNet gives only 5 facial points, and their jitter (about
  0.1 on the turn scale) is close to a real head turn on this camera (0.15 to
  0.34). This limits how strict the challenge can be. Dense landmarks (e.g. 68 or
  468 points) or a trained presentation-attack-detection model would separate
  real turns from photo movement far better.
- **Head-turn challenge:** a video can still pass if the random order happens to
  match within the 2 allowed attempts (about 1 in 4 in simulation); more steps or
  fewer attempts lower this but make it harder for genuine users. An attacker
  swapping between photos of the person facing left and right in the right order
  is not covered. Stronger defences need depth (3D/IR cameras) or
  texture analysis to detect screens and paper.
- Tested with one webcam and a small number of people; results are indicative,
  not a formal benchmark.

## Author

Asim Sohail, BS Cyber Security (IUB, 2021–2025)
