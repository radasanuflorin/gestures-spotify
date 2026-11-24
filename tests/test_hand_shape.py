"""The hand's shape from its 21 points: pointing sideways, turned like a knob, palm facing the camera."""

import math
from types import SimpleNamespace as P

from handgestures.hand import fingers_state, shape_of

ASPECT = 640 / 480

# An open right hand, palm to the camera, fingers up, in "real" proportions (x already in height units).
OPEN_UPRIGHT = [(0.0, 0.0), (-0.08, -0.05), (-0.13, -0.11), (-0.16, -0.17), (-0.19, -0.22),
                (-0.07, -0.20), (-0.08, -0.29), (-0.08, -0.35), (-0.08, -0.40),
                (0.0, -0.21), (0.0, -0.31), (0.0, -0.38), (0.0, -0.44),
                (0.06, -0.20), (0.07, -0.29), (0.07, -0.35), (0.07, -0.40),
                (0.11, -0.17), (0.13, -0.24), (0.14, -0.29), (0.15, -0.33)]
# Index out, other fingers curled back toward the palm.
POINTING_UP = [OPEN_UPRIGHT[i] for i in range(9)] + [
    (0.0, -0.21), (0.0, -0.26), (0.0, -0.19), (0.0, -0.15),
    (0.06, -0.20), (0.06, -0.24), (0.06, -0.18), (0.06, -0.14),
    (0.11, -0.17), (0.11, -0.21), (0.11, -0.16), (0.11, -0.13)]


def place(points, angle_deg=0.0, squash_x=1.0, at=(0.5, 0.6)):
    """Rotate (positive = clockwise on screen), squash sideways (a palm turned away), and convert to
    normalized image coordinates."""
    a = math.radians(angle_deg)
    out = []
    for x, y in points:
        x *= squash_x
        rx, ry = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        out.append(P(x=at[0] + rx / ASPECT, y=at[1] + ry))
    return out


def shape(lm):
    return shape_of(lm, fingers_state(lm), ASPECT)


def test_upright_open_palm_from_the_front():
    pointing, angle, frontal = shape(place(OPEN_UPRIGHT))
    assert pointing is None and abs(angle) < 5 and frontal > 0.7


def test_palm_seen_from_the_side_is_narrow():
    _, _, frontal = shape(place(OPEN_UPRIGHT, squash_x=0.3))
    assert frontal < 0.4


def test_angle_follows_the_turn():
    for turned in (30, 60, -45):
        _, angle, frontal = shape(place(OPEN_UPRIGHT, angle_deg=turned))
        assert abs(angle - turned) < 3
        assert frontal > 0.7  # turning in the image plane keeps the palm facing the camera


def test_index_pointing_right_and_left():
    assert shape(place(POINTING_UP, angle_deg=90))[0] == "right"
    assert shape(place(POINTING_UP, angle_deg=-90))[0] == "left"


def test_index_pointing_up_is_not_sideways():
    assert shape(place(POINTING_UP))[0] is None
