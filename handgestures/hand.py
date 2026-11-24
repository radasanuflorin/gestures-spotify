"""Hand detection with MediaPipe Gesture Recognizer: 21 hand points plus a trained open-palm / fist classifier."""

import logging
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)

# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky
FINGER_PIPS = (6, 10, 14, 18)
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
INDEX_TIP = 8
ALL_TIPS = (THUMB_TIP,) + FINGER_TIPS
FINGER_NAMES = ("deget mare", "aratator", "mijlociu", "inelar", "mic")


@dataclass
class Hand:
    landmarks: list  # 21 normalized landmarks (x, y in 0..1)
    fingers: list[bool]  # thumb, index, middle, ring, pinky: up?
    center: tuple[float, float]  # palm center
    gesture: str = "None"  # the classifier's label: Open_Palm, Closed_Fist, None, Thumb_Up, ...
    score: float = 0.0  # its confidence, 0..1
    handedness: str = ""  # "Left" or "Right", as MediaPipe sees it
    pointing: str | None = None  # "left" / "right": only the index finger out, pointing sideways
    angle: float = 0.0  # degrees the hand is turned from upright: + clockwise (to the right), - the other way
    frontal: float = 1.0  # palm width / palm length: about 0.8 seen from the front, 0.2-0.3 from the side

    @property
    def fingers_up(self) -> int:
        return sum(self.fingers)


def _dist2(a, b) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def fingers_state(lm) -> list[bool]:
    """Which fingers are up: thumb, index, middle, ring, pinky.

    A finger is up when its tip is farther from the wrist than its middle joint.
    """
    wrist = lm[WRIST]
    # Thumb: tip farther from the pinky base than the thumb's IP joint (works for both hands).
    thumb = _dist2(lm[THUMB_TIP], lm[PINKY_MCP]) > _dist2(lm[THUMB_IP], lm[PINKY_MCP]) * 1.1
    return [thumb] + [_dist2(lm[tip], wrist) > _dist2(lm[pip], wrist) for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]


def palm_center(lm) -> tuple[float, float]:
    return (
        (lm[WRIST].x + lm[INDEX_MCP].x + lm[PINKY_MCP].x) / 3,
        (lm[WRIST].y + lm[INDEX_MCP].y + lm[PINKY_MCP].y) / 3,
    )


DOMINANT = 1.5  # a direction counts as horizontal (or vertical) when it is this much more so than the other


def shape_of(lm, fingers: list[bool], aspect: float) -> tuple[str | None, float, float]:
    """(pointing, angle, frontal) from the hand points; see Hand. `aspect` is the image's width / height, so
    that directions are measured in real proportions (normalized x and y have different units)."""

    def vec(a, b):
        return (lm[b].x - lm[a].x) * aspect, lm[b].y - lm[a].y

    def length(v):
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    palm = vec(WRIST, MIDDLE_MCP)  # wrist to the base of the middle finger
    palm_len = length(palm) or 1e-6
    frontal = length(vec(INDEX_MCP, PINKY_MCP)) / palm_len

    pointing = None
    if fingers[1] and not any(fingers[2:]):  # index out, middle, ring and pinky folded
        finger = vec(INDEX_MCP, INDEX_TIP)
        if abs(finger[0]) >= DOMINANT * abs(finger[1]) and length(finger) >= 0.6 * palm_len:
            pointing = "right" if finger[0] > 0 else "left"

    # The image is mirrored, so it looks like the user's own view: fingers tilting right = clockwise.
    angle = math.degrees(math.atan2(palm[0], -palm[1]))
    return pointing, angle, frontal


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the model on first use (about 8 MB)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading the hand model to %s ...", path)
    partial = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the hand model ({e}). Download {url} to {path} by hand.") from e
    return path


class HandTracker:
    def __init__(self, model_path: Path = MODEL_PATH, min_confidence: float = 0.6, ignore_bottom: float = 0.0):
        """`ignore_bottom`: this bottom part of the image (the mouse and keyboard) is blacked out before the model
        looks, so a hand there is never found, and never picked instead of the hand making the gesture."""
        self.ignore_bottom = ignore_bottom
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr, timestamp_ms: int) -> Hand | None:
        timestamp_ms = max(timestamp_ms, self._last_ms + 1)  # MediaPipe needs strictly increasing times
        self._last_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # a new image: the caller's frame is left as it is
        if self.ignore_bottom > 0:
            rgb[int(rgb.shape[0] * (1 - self.ignore_bottom)):] = 0
        result = self._recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        fingers = fingers_state(lm)
        height, width = frame_bgr.shape[:2]
        pointing, angle, frontal = shape_of(lm, fingers, width / height)
        top = result.gestures[0][0] if result.gestures and result.gestures[0] else None
        side = result.handedness[0][0].category_name if result.handedness and result.handedness[0] else ""
        return Hand(
            landmarks=lm,
            fingers=fingers,
            center=palm_center(lm),
            gesture=top.category_name if top else "None",
            score=top.score if top else 0.0,
            handedness=side,
            pointing=pointing,
            angle=angle,
            frontal=frontal,
        )

    def close(self):
        self._recognizer.close()
