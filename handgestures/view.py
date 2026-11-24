"""Draws the hand analysis on a camera frame: the 21 points, gesture guides, status and progress."""

import math
import unicodedata

import cv2
from mediapipe.tasks.python import vision

from .gestures import FIST, MIDDLE, OPEN, OTHER, POINT_LEFT, POINT_RIGHT, pose_of
from .hand import ALL_TIPS, FINGER_NAMES, Hand

WHITE, BLACK, RED = (255, 255, 255), (0, 0, 0), (0, 0, 255)
GREEN, ORANGE, GREY, CYAN, YELLOW = (80, 220, 80), (0, 165, 255), (150, 150, 150), (255, 200, 0), (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
POSE_LABELS = {
    OPEN: ("PALMA DIN FATA", GREEN),
    FIST: ("PUMN", (0, 140, 255)),
    POINT_LEFT: ("DEGET SPRE STANGA", CYAN),
    POINT_RIGHT: ("DEGET SPRE DREAPTA", CYAN),
    MIDDLE: ("DEGETUL MIJLOCIU: inchide aplicatia", RED),
}
PROGRESS_LABELS = {"next": "piesa urmatoare: tine degetul", "previous": "piesa anterioara: tine degetul",
                   "pause": "pauza: tine pumnul", "play": "play: tine palma",
                   "quit": "inchidere: tine degetul mijlociu",
                   "knob_up": "volum +: roteste spre dreapta", "knob_down": "volum -: roteste spre stanga"}


def ascii_text(text: str) -> str:
    """OpenCV fonts have no diacritics: "Următoarea" -> "Urmatoarea"."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _text(frame, text, org, scale=0.6, color=WHITE, thickness=2):
    """Text with a dark outline, readable on any background.

    The outline is the same text in black, shifted around it: drawing it thicker instead would also widen
    the letter spacing, and a long outline would drift away from its text.
    """
    x, y = org
    r = 1 if thickness == 1 else 2
    for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, r), (-r, r), (r, -r)):
        cv2.putText(frame, text, (x + dx, y + dy), FONT, scale, BLACK, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hand(frame, hand: Hand, trail=()):
    """Skeleton, fingertips (green = up, grey = down), palm center and its recent path."""
    h, w = frame.shape[:2]

    def px(p):
        return int(p.x * w), int(p.y * h)

    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, px(hand.landmarks[conn.start]), px(hand.landmarks[conn.end]), (230, 230, 230), 2, cv2.LINE_AA)
    for i, p in enumerate(hand.landmarks):
        if i not in ALL_TIPS:
            cv2.circle(frame, px(p), 4, RED, -1, cv2.LINE_AA)
    for tip, up in zip(ALL_TIPS, hand.fingers):
        cv2.circle(frame, px(hand.landmarks[tip]), 11, GREEN if up else GREY, -1, cv2.LINE_AA)
        cv2.circle(frame, px(hand.landmarks[tip]), 11, BLACK, 2, cv2.LINE_AA)

    points = [(int(x * w), int(y * h)) for x, y in trail]
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], CYAN, 1 + 5 * i // len(points), cv2.LINE_AA)
    cv2.drawMarker(frame, (int(hand.center[0] * w), int(hand.center[1] * h)), CYAN, cv2.MARKER_CROSS, 22, 3)


HUD_HEIGHT = 64  # the black status band at the top


def draw_guides(frame, ignore_bottom: float):
    """The gestures' guides, drawn faintly over the camera: where the hand is ignored (mouse and keyboard), the
    two pointing directions for the track, and the knob the open hand turns for the volume."""
    h, w = frame.shape[:2]
    usable = int(h * (1 - ignore_bottom))  # bottom of the area the gestures use
    guide = (235, 235, 235)
    if ignore_bottom > 0:
        layer = frame.copy()
        cv2.rectangle(layer, (0, usable), (w, h), (20, 20, 90), -1)
        cv2.addWeighted(layer, 0.35, frame, 0.65, 0, dst=frame)
        cv2.line(frame, (0, usable), (w, usable), (60, 60, 220), 2)
        _text(frame, "ignorat: mouse / tastatura", (w - 250, usable + 22), 0.5, guide, 1)

    # Track: point the index finger to a side.
    mid_y = (HUD_HEIGHT + usable) // 2
    cv2.arrowedLine(frame, (70, mid_y), (20, mid_y), guide, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(frame, (w - 70, mid_y), (w - 20, mid_y), guide, 2, cv2.LINE_AA, tipLength=0.4)
    _text(frame, "deget: inapoi", (12, mid_y - 14), 0.45, guide, 1)
    _text(frame, "deget: urmatoarea", (w - 160, mid_y - 14), 0.45, guide, 1)

    # Volume: the open hand turns like a knob.
    cx, cy, r = w // 2, HUD_HEIGHT + 55, 32
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 200, 340, guide, 2, cv2.LINE_AA)  # the knob's top arc
    end = (cx + int(r * math.cos(math.radians(340))), cy + int(r * math.sin(math.radians(340))))
    start = (cx + int(r * math.cos(math.radians(200))), cy + int(r * math.sin(math.radians(200))))
    cv2.arrowedLine(frame, (end[0] - 6, end[1] - 8), end, guide, 2, cv2.LINE_AA, tipLength=1.0)
    cv2.arrowedLine(frame, (start[0] + 6, start[1] - 8), start, guide, 2, cv2.LINE_AA, tipLength=1.0)
    _text(frame, "+", (end[0] + 8, end[1] + 4), 0.6, guide, 2)
    _text(frame, "-", (start[0] - 20, start[1] + 4), 0.6, guide, 2)
    _text(frame, "volum: roteste palma", (cx - 80, cy + 30), 0.45, guide, 1)


LIGHT_WARNINGS = {
    "dark": "lumina slaba: aprinde o lumina pentru detectie buna",
    "covered": "imagine neagra: camera e acoperita sau are clapeta inchisa?",
}


def draw_hud(frame, hand, *, active: bool, fps: float, light: str | None, progress, action: str, action_age: float,
             ignored: bool = False, pose: str | None = None):
    """Status, pose and classifier output, the gesture under way, warnings, and the last gesture."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, HUD_HEIGHT), BLACK, -1)
    status, color = ("ACTIV", (0, 255, 0)) if active else ("EXERSARE (Spotify nu e activ)", ORANGE)
    _text(frame, status, (10, 24), 0.6, color)
    _text(frame, f"{fps:.0f} fps", (w - 80, 24), 0.55, GREY if fps >= 20 else ORANGE)
    if hand and ignored:
        _text(frame, "mana in zona ignorata (mouse / tastatura)", (10, 52), 0.55, GREY)
    elif hand:
        pose = pose or pose_of(hand)
        if pose == OTHER and hand.fingers_up >= 4:
            label, pose_color = "PALMA DIN PROFIL: intoarce-o spre camera", ORANGE
        else:
            label, pose_color = POSE_LABELS.get(pose, ("alta pozitie", (200, 200, 200)))
            if pose == OPEN:
                label += f" {hand.angle:+.0f} grade"
        model = f"model: {hand.gesture} {hand.score:.0%}" if hand.gesture not in ("", "None") else "model: nesigur"
        _text(frame, f"{label}  {hand.fingers_up} degete   {model}", (10, 52), 0.55, pose_color)
        up = [name for name, is_up in zip(FINGER_NAMES, hand.fingers) if is_up]
        _text(frame, ", ".join(up) or "niciun deget ridicat", (10, h - 14), 0.5)
    else:
        _text(frame, "nicio mana", (10, 52), 0.55, RED)

    if progress and progress[1] >= 0.15:
        name, fraction = progress
        bar_w = int((w - 40) * fraction)
        cv2.rectangle(frame, (20, h - 52), (w - 20, h - 38), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 52), (20 + bar_w, h - 38), YELLOW if fraction < 1 else GREEN, -1)
        _text(frame, PROGRESS_LABELS[name], (20, h - 60), 0.5, YELLOW)

    if light:
        _text(frame, LIGHT_WARNINGS[light], (10, 90), 0.5, ORANGE if light == "dark" else RED)

    if action and action_age < 1.5:
        action = ascii_text(action)
        size = cv2.getTextSize(action, FONT, 1.3, 3)[0]
        _text(frame, action, ((w - size[0]) // 2, h // 2), 1.3, YELLOW, 3)

