import pytest

from handgestures.gestures import FIST, NONE, OPEN, OTHER, POINT_LEFT, POINT_RIGHT, pose_of
from handgestures.hand import Hand


def hand(fingers_up: int, gesture: str = "None", score: float = 0.0, **shape) -> Hand:
    return Hand(landmarks=[], fingers=[i < fingers_up for i in range(5)], center=(0.5, 0.5), gesture=gesture,
                score=score, **shape)


def test_no_hand():
    assert pose_of(None) == NONE


@pytest.mark.parametrize(
    "h, pose",
    [
        (hand(5, "Open_Palm", 0.9), OPEN),
        (hand(0, "Closed_Fist", 0.9), FIST),
        # the confident classifier wins over a wrong finger count
        (hand(3, "Open_Palm", 0.8), OPEN),
        (hand(1, "Closed_Fist", 0.8), FIST),
        (hand(5, "Victory", 0.9), OTHER),
        (hand(1, "Pointing_Up", 0.9), OTHER),
        # unsure classifier: the finger count decides
        (hand(5, "None", 0.7), OPEN),
        (hand(4, "Open_Palm", 0.3), OPEN),
        (hand(0, "None", 0.6), FIST),
        (hand(2, "None", 0.6), OTHER),
        # ...but never against the classifier's own (weak) label
        (hand(5, "Closed_Fist", 0.4), OTHER),
        (hand(0, "Open_Palm", 0.4), OTHER),
    ],
)
def test_pose(h, pose):
    assert pose_of(h) == pose


@pytest.mark.parametrize(
    "h, pose",
    [
        (hand(2, "None", 0.0, pointing="right"), POINT_RIGHT),
        (hand(2, "Pointing_Up", 0.6, pointing="left"), POINT_LEFT),  # shape wins over the classifier
        (hand(5, "Open_Palm", 0.9, angle=60), OPEN),  # a turned open hand is still open (the volume knob)
        (hand(5, "Open_Palm", 0.9, frontal=0.3), OTHER),  # open palm seen from the side: no play
        (hand(5, "None", 0.7, frontal=0.3), OTHER),
        (hand(5, "Open_Palm", 0.9, frontal=0.7), OPEN),
    ],
)
def test_shape_poses(h, pose):
    assert pose_of(h) == pose


def test_frontal_threshold_is_adjustable():
    assert pose_of(hand(5, "Open_Palm", 0.9, frontal=0.45), frontal_min=0.4) == OPEN


@pytest.mark.parametrize(
    "fingers, pose",
    [
        ([False, False, True, False, False], "middle_finger"),
        ([True, False, True, False, False], "middle_finger"),  # thumb out too
        ([False, True, True, False, False], OTHER),  # two fingers: not it
    ],
)
def test_middle_finger(fingers, pose):
    # the classifier has no label for it: it may call it a fist, or be unsure
    h = Hand(landmarks=[], fingers=fingers, center=(0.5, 0.5), gesture="None", score=0.0)
    assert pose_of(h) == pose


def test_middle_finger_wins_over_a_confident_fist_label():
    h = Hand(landmarks=[], fingers=[False, False, True, False, False], center=(0.5, 0.5), gesture="Closed_Fist",
             score=0.8)
    assert pose_of(h) == "middle_finger"
