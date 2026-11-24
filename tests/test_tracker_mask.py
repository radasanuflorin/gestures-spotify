"""The model never sees the ignored bottom of the image."""

from types import SimpleNamespace

import numpy as np

from handgestures.hand import HandTracker


def tracker_seeing(ignore_bottom):
    """A tracker whose model only records the image it is given."""
    t = object.__new__(HandTracker)
    t.ignore_bottom, t._last_ms, t.seen = ignore_bottom, -1, []

    def recognize(image, _ms):
        t.seen.append(image.numpy_view().copy())
        return SimpleNamespace(hand_landmarks=[], gestures=[], handedness=[])

    t._recognizer = SimpleNamespace(recognize_for_video=recognize)
    return t


def test_bottom_zone_is_black_for_the_model_but_not_for_the_caller():
    frame = np.full((480, 640, 3), 200, np.uint8)
    t = tracker_seeing(0.25)
    assert t.detect(frame, 0) is None
    seen = t.seen[0]
    assert seen[:360].min() == 200 and seen[360:].max() == 0  # above the line kept, below it black
    assert frame.min() == 200  # the frame shown in the preview is untouched


def test_no_zone_leaves_the_image_whole():
    t = tracker_seeing(0.0)
    t.detect(np.full((480, 640, 3), 200, np.uint8), 0)
    assert t.seen[0].min() == 200
