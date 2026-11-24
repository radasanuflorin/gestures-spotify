from handgestures.gestures import FIST, NONE, OPEN, OTHER, PoseHold

FPS = 30


def run(poses, fps=FPS, positions=None):
    d = PoseHold()
    positions = positions or [(0.5, 0.5)] * len(poses)
    return [(round(i / fps, 2), e) for i, (p, pos) in enumerate(zip(poses, positions)) if (e := d.update(p, i / fps, pos))]


def events(*args, **kwargs):
    return [e for _, e in run(*args, **kwargs)]


def test_fist_held_two_seconds_pauses():
    assert run([FIST] * 70) == [(2.0, "pause")]


def test_open_palm_held_two_seconds_plays():
    assert events([OPEN] * 70) == ["play"]


def test_shorter_hold_does_nothing():
    assert events([FIST] * 50 + [NONE] * 30) == []


def test_fires_once_per_hold():
    assert events([FIST] * 200) == ["pause"]


def test_new_hold_after_changing_pose():
    assert events([FIST] * 70 + [OPEN] * 70) == ["pause", "play"]


def test_same_at_10_fps():
    assert events([FIST] * 25, fps=10) == ["pause"]


def test_one_misread_frame_does_not_restart_the_hold():
    assert run([FIST] * 30 + [OPEN] + [FIST] * 35) == [(2.0, "pause")]


def test_brief_loss_of_the_hand_does_not_restart_the_hold():
    assert run([FIST] * 30 + [NONE] * 5 + [FIST] * 30) == [(2.0, "pause")]


def test_long_loss_restarts_the_hold():
    assert events([FIST] * 40 + [NONE] * 20 + [FIST] * 40) == []


def test_other_poses_do_nothing():
    assert events([OTHER] * 100) == []


def test_moving_open_hand_does_not_play():
    # an open palm crossing the camera is never held in place
    positions = [(0.1 + 0.8 * i / 70, 0.5) for i in range(70)]
    assert events([OPEN] * 70, positions=positions) == []


def test_slight_wobble_still_counts():
    positions = [(0.5 + 0.02 * (i % 3), 0.5) for i in range(70)]
    assert events([FIST] * 70, positions=positions) == ["pause"]


def test_progress_fills_up_while_holding():
    d = PoseHold()
    for i in range(31):
        d.update(FIST, i / FPS, (0.5, 0.5))
    name, fraction = d.progress
    assert name == "pause" and 0.45 < fraction < 0.55


def test_switching_sign_counts_from_when_the_new_one_appeared():
    # pumn 2.1 s, then palm 2.05 s: the palm's hold is timed from its first frame, not after the doubt period
    assert events([FIST] * 63 + [OPEN] * 62) == ["pause", "play"]
