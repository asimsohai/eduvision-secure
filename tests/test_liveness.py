from eduvision.evaluation import markdown_table, summarise
from eduvision.liveness import BlinkCounter

O, C = True, False  # eyes open / closed


def feed(counter, frames):
    for f in frames:
        counter.update(f)
    return counter.blinks


def test_normal_blink_counts():
    assert feed(BlinkCounter(), [O, O, O, C, C, C, O]) == 1


def test_static_photo_never_blinks():
    assert feed(BlinkCounter(), [O] * 200) == 0


def test_single_frame_flicker_is_ignored():
    # the eye detector missing one frame must not count as a blink
    assert feed(BlinkCounter(min_closed=2), [O, O, O, C, O, O, O, C, O]) == 0


def test_long_closure_is_not_a_blink():
    # eyes shut (or face turned) for a long time is not a blink
    assert feed(BlinkCounter(max_closed=8), [O, O, O] + [C] * 20 + [O]) == 0


def test_needs_open_eyes_before_closing():
    # a photo with closed eyes "opening" is not a blink
    assert feed(BlinkCounter(min_open=2), [C, C, C, O]) == 0


def test_two_blinks():
    assert feed(BlinkCounter(), [O, O, C, C, O, O, C, C, O]) == 2


def test_reset():
    c = BlinkCounter()
    feed(c, [O, O, C, C, O])
    c.reset()
    assert c.blinks == 0


def _rows(liveness, label, accepted_flags):
    return [
        {"liveness": liveness, "label": label, "accepted": str(a),
         "seconds_to_accept": "1.5" if a else ""}
        for a in accepted_flags
    ]


def test_summary_metrics():
    rows = (_rows("off", "print", [1] * 10)
            + _rows("blink", "print", [0] * 9 + [1])
            + _rows("blink", "genuine", [1] * 8 + [0] * 2))
    s = {(r["liveness"], r["label"]): r for r in summarise(rows)}
    assert s[("off", "print")]["error_rate"] == 1.0      # APCER 100%
    assert s[("blink", "print")]["error_rate"] == 0.1       # APCER 10%
    assert s[("blink", "genuine")]["metric"] == "BPCER"
    assert s[("blink", "genuine")]["error_rate"] == 0.2     # 2 of 10 rejected
    assert "| BLINK | print | 10 | 1 | APCER | 10% |" in markdown_table(summarise(rows))


# --- head-turn challenge ---------------------------------------------------

import random

import numpy as np

from eduvision.face_engine import Face
from eduvision.liveness import ChallengeLivenessChecker, classify_pose, head_turn_offset


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_checker(steps=3, seed=1):
    clock = FakeClock()
    chk = ChallengeLivenessChecker(steps=steps, hold=3, step_seconds=5,
                                   threshold=0.18, rng=random.Random(seed), clock=clock)
    return chk, clock


def hold(chk, pose, frames=3, clock=None, dt=0.1):
    for _ in range(frames):
        if clock:
            clock.t += dt
        chk.update_pose(pose)


def fake_face(eye1_x, eye2_x, nose_x):
    raw = np.zeros(15, np.float32)
    raw[4], raw[6], raw[8] = eye1_x, eye2_x, nose_x
    return Face(box=(0, 0, 100, 100), score=0.99, raw=raw)


def test_head_turn_offset_and_pose():
    assert classify_pose(head_turn_offset(fake_face(40, 80, 60)), 0.18) == "center"
    assert classify_pose(head_turn_offset(fake_face(40, 80, 72)), 0.18) == "left"
    assert classify_pose(head_turn_offset(fake_face(40, 80, 48)), 0.18) == "right"
    # same face photographed smaller (further away) gives the same answer
    assert classify_pose(head_turn_offset(fake_face(20, 40, 36)), 0.18) == "left"


def test_correct_sequence_passes():
    chk, clock = make_checker()
    for step in list(chk.steps):
        hold(chk, "center", clock=clock)
        hold(chk, step, clock=clock)
    assert chk.is_live()


def test_static_photo_never_passes():
    chk, clock = make_checker()
    for _ in range(300):                       # 30 seconds of a frontal photo
        hold(chk, "center", frames=1, clock=clock)
    assert not chk.is_live()


def test_wrong_direction_generates_new_challenge():
    chk, clock = make_checker()
    wrong = "right" if chk.steps[0] == "left" else "left"
    hold(chk, "center", clock=clock)
    hold(chk, wrong, clock=clock)
    assert chk.failures == 1 and chk.index == 0 and not chk.is_live()


def test_too_slow_times_out():
    chk, clock = make_checker()
    hold(chk, "center", clock=clock)
    clock.t += 6                                # longer than step_seconds
    hold(chk, chk.steps[0], clock=clock)
    assert chk.failures == 1 and not chk.is_live()


def test_must_return_to_center_between_turns():
    chk, clock = make_checker(seed=3)
    chk.steps = ["left", "left", "left"]
    hold(chk, "center", clock=clock)
    hold(chk, "left", frames=20, clock=clock)   # staying turned does not count 3 times
    assert chk.index == 1 and not chk.is_live()


def test_replayed_fixed_video_rarely_matches():
    # a video that always does LEFT, RIGHT, LEFT matches a random 3-step challenge
    # only when the challenge happens to be the same: about 1 in 8
    rng = random.Random(0)
    matches = sum(
        ChallengeLivenessChecker(steps=3, rng=rng).steps == ["left", "right", "left"]
        for _ in range(4000)
    )
    assert 0.09 < matches / 4000 < 0.16


def test_summary_maps_old_on_rows_to_blink():
    rows = _rows("on", "screen", [1, 0])
    assert summarise(rows)[0]["liveness"] == "blink"


def feed_offset(chk, offset, frames, clock, dt=0.1):
    for _ in range(frames):
        clock.t += dt
        chk.update_offset(offset)


def test_off_centre_camera_is_calibrated():
    # laptop webcam: the person's "straight" pose already reads 0.12
    chk, clock = make_checker()
    feed_offset(chk, 0.12, 3, clock)                  # calibration
    assert abs(chk.baseline - 0.12) < 1e-6
    for step in list(chk.steps):
        feed_offset(chk, 0.12, 3, clock)              # straight
        feed_offset(chk, 0.12 + (0.25 if step == "left" else -0.25), 3, clock)
    assert chk.is_live()


def test_tilted_photo_cannot_pass_after_calibration():
    # a tilted flat photo changes the offset by < 0.1 (measured), not enough
    chk, clock = make_checker()
    for i in range(300):
        feed_offset(chk, 0.05 + 0.09 * ((i // 10) % 2), 1, clock)
    assert not chk.is_live()


def test_attempt_limit_locks_out_brute_force():
    clock = FakeClock()
    chk = ChallengeLivenessChecker(steps=3, hold=2, step_seconds=6, threshold=0.12,
                                   rng=random.Random(5), clock=clock,
                                   max_attempts=2, lockout_seconds=30)
    for _ in range(2):                                   # two wrong attempts
        wrong = "right" if chk.steps[0] == "left" else "left"
        hold(chk, "center", frames=2, clock=clock)
        hold(chk, wrong, frames=2, clock=clock)
    assert chk.locked
    # even a perfect sequence is ignored while locked
    for step in list(chk.steps):
        hold(chk, "center", frames=2, clock=clock)
        hold(chk, step, frames=2, clock=clock)
    assert not chk.is_live()
    assert "Try again" in chk.prompt
    # after the lockout a genuine user can pass (a fresh challenge is generated)
    clock.t += 31
    hold(chk, "center", frames=1, clock=clock)
    assert not chk.locked
    for step in list(chk.steps):
        hold(chk, "center", frames=2, clock=clock)
        hold(chk, step, frames=2, clock=clock)
    assert chk.is_live()


def test_looping_video_cannot_brute_force_within_one_session():
    # attacker cycles LEFT/RIGHT forever; with the limit, success needs the
    # random order to match within 2 attempts, so most sessions fail
    wins = 0
    for seed in range(500):
        clock = FakeClock()
        chk = ChallengeLivenessChecker(steps=3, hold=2, step_seconds=6, threshold=0.12,
                                       rng=random.Random(seed), clock=clock,
                                       max_attempts=2, lockout_seconds=30)
        pattern = ["left", "right"]
        for i in range(60):                              # ~15 s of video
            hold(chk, "center", frames=2, clock=clock)
            hold(chk, pattern[i % 2], frames=2, clock=clock)
        wins += chk.is_live()
    assert wins / 500 < 0.35


def test_eval_clear_keeps_other_modes(tmp_path):
    import csv as _csv
    from eduvision.evaluation import FIELDS, clear_mode, read_results
    path = tmp_path / "e.csv"
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for mode in ("off", "challenge", "challenge", "blink"):
            w.writerow({k: "" for k in FIELDS} | {"liveness": mode})
    assert clear_mode(path, "challenge") == 2
    assert [r["liveness"] for r in read_results(path)] == ["off", "blink"]
