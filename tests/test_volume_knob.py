from handgestures.gestures import FIST, NONE, OPEN, VolumeKnob

FPS = 30


def run(poses_and_angles, knob=None):
    k = knob or VolumeKnob()
    return [e for i, (pose, a) in enumerate(poses_and_angles) if (e := k.update(pose, i / FPS, a))]


def turn(a0, a1, frames, pose=OPEN, still_before=15, still_after=5):
    """The open hand held at angle a0, turned to a1 over `frames` frames, then held."""
    moving = [(pose, a0 + (a1 - a0) * i / (frames - 1)) for i in range(frames)]
    return [(pose, a0)] * still_before + moving + [(pose, a1)] * still_after


def test_clockwise_turns_the_volume_up_step_by_step():
    events = run(turn(0, 47, 20))
    assert events == ["volume_up"] * 3  # 47 / 15


def test_counterclockwise_turns_it_down():
    assert run(turn(0, -47, 20)) == ["volume_down"] * 3


def test_bringing_the_hand_back_does_not_lower_the_volume():
    # turn right, then slowly back to neutral: only the turn away counts
    assert run(turn(0, 32, 15) + turn(32, 0, 60, still_before=0)) == ["volume_up"] * 2


def test_turning_again_after_coming_back_adds_more():
    # the ratchet: turn, bring back, turn again
    events = run(turn(0, 32, 15) + turn(32, 0, 20, still_before=0) + turn(0, 32, 15, still_before=0))
    assert events == ["volume_up"] * 4


def test_past_neutral_the_other_way_lowers_it():
    events = run(turn(0, 32, 15) + turn(32, -32, 30, still_before=0))
    assert events == ["volume_up"] * 2 + ["volume_down"] * 2


def test_trembling_on_a_step_edge_fires_once():
    assert run(turn(0, 15, 10, still_after=0) + [(OPEN, 15 + 3 * (-1) ** i) for i in range(60)]) == ["volume_up"]


def test_hand_coming_in_already_tilted_starts_at_zero():
    # the knob's zero is the angle once the hand has settled, not upright
    assert run([(OPEN, 40)] * 60) == []


def test_jitter_does_nothing():
    assert run([(OPEN, 5 * (-1) ** i) for i in range(90)]) == []


def test_fist_turning_does_nothing():
    assert run(turn(0, 60, 20, pose=FIST)) == []


def test_turning_while_the_hand_settles_does_not_count():
    assert run(turn(0, 40, 6, still_before=0, still_after=30)) == []


def test_new_hand_starts_a_new_knob():
    # turn right, drop the hand, raise it upright again, turn right again: two separate turns add up
    events = run(turn(0, 32, 15) + [(NONE, None)] * 20 + turn(0, 32, 15))
    assert events == ["volume_up"] * 4


def test_busy_while_turning_and_shortly_after():
    k = VolumeKnob()
    run(turn(0, 20, 10), knob=k)  # ends at frame 29 (~0.97 s) with one step done, hand still turned
    assert k.busy(1.0) and k.busy(9.0)  # turned away from neutral: not holding a sign
    k.update(OPEN, 9.0, 0.0)
    k.update(OPEN, 9.03, 0.0)
    k.update(OPEN, 9.06, 0.0)  # back to neutral (after smoothing)
    assert not k.busy(9.1)
