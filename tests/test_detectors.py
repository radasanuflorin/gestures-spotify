"""All the detectors together, on simulated hands, as the main loop runs them."""

from handgestures import config
from handgestures.detectors import Detectors
from handgestures.hand import Hand

FPS = 30


def open_hand(x=0.5, y=0.5, **shape):
    return Hand(landmarks=[], fingers=[True] * 5, center=(x, y), gesture="Open_Palm", score=0.9, **shape)


def fist(x=0.5, y=0.5):
    return Hand(landmarks=[], fingers=[False] * 5, center=(x, y), gesture="Closed_Fist", score=0.9)


def pointing(direction, x=0.5, y=0.5):
    return Hand(landmarks=[], fingers=[False, True, False, False, False], center=(x, y), gesture="None",
                pointing=direction)


def turned(angle, x=0.5, y=0.5):
    return open_hand(x, y, angle=angle)


def run(hands):
    d = Detectors(config.DEFAULTS)
    return [e for i, h in enumerate(hands) for e in d.update(h, i / FPS)]


def seconds(s):
    return int(s * FPS)


def test_pause_then_play_by_holding_the_signs():
    assert run([fist(y=0.52)] * seconds(2.3) + [open_hand()] * seconds(2.3)) == ["pause", "play"]


def test_palm_from_the_side_does_not_play():
    assert run([open_hand(frontal=0.3)] * seconds(3)) == []


def test_pointing_right_held_one_second_is_next():
    assert run([pointing("right")] * seconds(1.3)) == ["next"]


def test_pointing_left_is_previous():
    assert run([pointing("left")] * seconds(1.3)) == ["previous"]


def test_pointing_shortly_does_nothing():
    assert run([pointing("right")] * seconds(0.6) + [None] * 20) == []


def test_two_tracks_need_two_points():
    hands = [pointing("right")] * seconds(1.3) + [None] * 20 + [pointing("right")] * seconds(1.3)
    assert run(hands) == ["next", "next"]


def knob_turn(a0, a1, frames=20):
    return [turned(a0)] * 15 + [turned(a0 + (a1 - a0) * i / (frames - 1)) for i in range(frames)] + [turned(a1)] * 5


def test_turning_the_open_hand_clockwise_turns_the_volume_up():
    assert run(knob_turn(0, 47)) == ["volume_up"] * 3


def test_turning_it_the_other_way_turns_it_down():
    assert run(knob_turn(0, -47)) == ["volume_down"] * 3


def test_turning_the_knob_does_not_play():
    # an open palm turned slowly for 3 s, then held: it never counts as holding the sign for play
    hands = [turned(0)] * 15 + [turned(50 * i / 89) for i in range(90)] + [turned(50)] * 20
    assert "play" not in run(hands)


def test_play_works_once_the_hand_is_back_upright():
    hands = knob_turn(0, 32) + [turned(32 - 32 * i / 14) for i in range(15)] + [turned(0)] * seconds(3.5)
    assert run(hands) == ["volume_up", "volume_up", "play"]


def test_a_hand_held_turned_does_not_play():
    assert run(knob_turn(0, 32) + [turned(32)] * seconds(4)) == ["volume_up", "volume_up"]


def test_hand_leaving_the_image_does_nothing():
    assert run([open_hand()] * 10 + [None] * 30 + [fist()] * 10) == []


def side(h, handedness):
    h.handedness = handedness
    return h


def test_jumping_to_the_other_hand_does_not_turn_the_knob():
    # Two open hands in view at different angles: the tracker switches from one to the other.
    hands = [side(turned(-20, x=0.2), "Left") for _ in range(20)] + [side(turned(30, x=0.8), "Right") for _ in range(20)]
    assert run(hands) == []


def test_hand_in_the_bottom_zone_is_ignored():
    # a fist resting on the mouse, low in the image, held for a long time
    d = Detectors(config.DEFAULTS)
    events = [e for i in range(120) for e in d.update(fist(0.5, 0.9), i / FPS)]
    assert events == [] and d.ignored


def test_pointing_in_the_bottom_zone_is_ignored():
    assert run([pointing("right", y=0.9)] * seconds(2)) == []


def test_play_pause_hold_is_1_2_seconds():
    assert run([fist(y=0.52)] * seconds(1.1)) == []
    assert run([fist(y=0.52)] * seconds(1.3)) == ["pause"]


def test_preview_shows_the_play_bar_while_holding_the_palm():
    d = Detectors(config.DEFAULTS)
    for i in range(seconds(0.8)):
        d.update(open_hand(), i / FPS)
    name, fraction = d.progress
    assert name == "play" and 0.5 < fraction < 0.8


def run_timed(hands):
    d = Detectors(config.DEFAULTS)
    return [(round(i / FPS, 2), e) for i, h in enumerate(hands) for e in d.update(h, i / FPS)]


def test_next_sign_waits_for_the_pause_after_an_action():
    # pause at 1.2 s; pointing right straight after: its 1 s hold only counts from 2.2 s, so next at ~3.2 s
    hands = [fist(y=0.52)] * seconds(1.25) + [pointing("right")] * seconds(2.5)
    timed = run_timed(hands)
    assert [e for _, e in timed] == ["pause", "next"]
    assert timed[1][0] >= 3.15


def test_holding_the_finger_after_next_does_not_repeat():
    assert run([pointing("right")] * seconds(4)) == ["next"]


def test_one_knob_turn_keeps_stepping():
    assert run(knob_turn(0, 47)) == ["volume_up"] * 3


def test_turning_the_knob_back_right_after_does_not_lower_the_volume():
    # turn right, then straight back to upright (bringing the hand back) within the pause
    hands = knob_turn(0, 32, frames=12) + [turned(32 - 34 * i / 8) for i in range(9)] + [turned(-2)] * 10
    assert run(hands) == ["volume_up", "volume_up"]


def middle_finger():
    return Hand(landmarks=[], fingers=[False, False, True, False, False], center=(0.5, 0.5), gesture="None")


def test_middle_finger_held_one_second_quits():
    assert run([middle_finger()] * seconds(1.2)) == ["quit"]


def test_middle_finger_flash_does_not_quit():
    assert run([middle_finger()] * seconds(0.5) + [None] * 30) == []


def test_quit_can_be_turned_off():
    import copy
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["quit"]["enabled"] = False
    d = Detectors(cfg)
    assert [e for i in range(seconds(2)) for e in d.update(middle_finger(), i / FPS)] == []
