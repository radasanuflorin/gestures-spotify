"""Camera check without Spotify: the webcam with the hand analysis, as the app sees it. Press Q to quit.

Use it to check the light, the distance and how the gestures are recognized.
"""

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # before OpenCV loads: see camera.py

import time
from collections import deque

import cv2

from . import config
from .camera import Camera, brighten, light_problem
from .detectors import MAX_FRAME_GAP, TRAIL_SECONDS, Detectors
from .hand import HandTracker
from .view import draw_guides, draw_hand, draw_hud

TITLE = "gestures-spotify: camera (Q = iesire)"


def main():
    cfg = config.load()
    cam = cfg["camera"]
    camera = Camera(cam["index"], cam["width"], cam["height"], cam["backend"])
    tracker = HandTracker(min_confidence=cfg["hand"]["min_confidence"], ignore_bottom=cfg["hand"]["ignore_bottom"])
    detectors = Detectors(cfg)
    trail: deque = deque()
    start = last = time.monotonic()
    fps, last_action, last_action_at = 0.0, "", -1e9
    try:
        while True:
            frame = camera.read()
            now = time.monotonic()
            if frame is None or (camera.duplicate and now - last < MAX_FRAME_GAP):
                continue
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            light = light_problem(frame)
            if cfg["camera"]["brighten"]:
                frame = brighten(frame)
            hand = tracker.detect(frame, int((now - start) * 1000))
            for event in detectors.update(hand, now):
                last_action, last_action_at = event, now
            if hand:
                trail.append((now, hand.center))
            while trail and (not hand or now - trail[0][0] > TRAIL_SECONDS):
                trail.popleft()
            draw_guides(frame, cfg["hand"]["ignore_bottom"])
            if hand:
                draw_hand(frame, hand, [c for _, c in trail])
            draw_hud(frame, hand, active=False, fps=fps, light=light, progress=detectors.progress,
                     action=last_action, action_age=now - last_action_at,
                     ignored=detectors.ignored, pose=detectors.pose)
            cv2.imshow(TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        camera.off()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
