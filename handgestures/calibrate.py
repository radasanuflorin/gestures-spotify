"""Guided gesture test: tells you on screen which gesture to make, then reports how well it was recognized.

The report says, for each step, how often the hand was found and its pose recognized, which gestures fired
(and whether any fired by accident), and suggests values for config.toml. It also compares detection on the
camera image as it is and brightened, to show whether a dim room is the problem.

Nothing is sent anywhere and no image is saved: only numbers, in %LOCALAPPDATA%\\gestures-spotify\\calibration\\.
Spotify is not controlled. Press Q to stop early.
"""

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # before OpenCV loads: see camera.py

import csv
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import config
from .camera import COVERED_BRIGHTNESS, DARK_BRIGHTNESS, Camera, brighten, light_problem
from .detectors import MAX_FRAME_GAP, TRAIL_SECONDS, Detectors
from .gestures import FIST, OPEN, POINT_LEFT, POINT_RIGHT, pose_of
from .hand import HandTracker
from .view import _text, draw_guides, draw_hand, draw_hud

TITLE = "gestures-spotify: test ghidat (Q = stop)"
OUT_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "gestures-spotify" / "calibration"


@dataclass
class Step:
    name: str
    seconds: float
    prompt: str
    expected: dict = field(default_factory=dict)  # event -> how many times it should fire
    pose: str | None = None  # the pose that should be seen (for the recognition rate)


STEPS = [
    Step("pregatire", 6, "Pregatire: pune mana in fata camerei, la 40-80 cm"),
    Step("palma", 8, "Tine PALMA DESCHISA cu fata spre camera, nemiscata", {"play": 1}, pose=OPEN),
    Step("pumn", 8, "Tine PUMNUL strans spre camera, nemiscat", {"pause": 1}, pose=FIST),
    Step("play-pauza", 10, "Pumn 2 s -> palma 2 s -> pumn 2 s -> palma 2 s", {"pause": 2, "play": 2}),
    Step("dreapta", 12, "Arata cu ARATATORUL spre DREAPTA 1 s, apoi lasa mana; de 3 ori", {"next": 3},
         pose=POINT_RIGHT),
    Step("stanga", 12, "Arata cu ARATATORUL spre STANGA 1 s, apoi lasa mana; de 3 ori", {"previous": 3},
         pose=POINT_LEFT),
    Step("sus", 10, "Palma spre camera: ROTESTE-O ~45 grade spre DREAPTA, apoi revino; de 2 ori",
         {"volume_up": 6}, pose=OPEN),
    Step("jos", 10, "Palma spre camera: ROTESTE-O ~45 grade spre STANGA, apoi revino; de 2 ori",
         {"volume_down": 6}, pose=OPEN),
    Step("normal", 15, "Fara gesturi: misca-te normal, ia mouse-ul, scarpina-te...", {}),
]


def main():
    cfg = config.load()
    cam = cfg["camera"]
    camera = Camera(cam["index"], cam["width"], cam["height"], cam["backend"])
    tracker = HandTracker(min_confidence=cfg["hand"]["min_confidence"], ignore_bottom=cfg["hand"]["ignore_bottom"])
    tracker_bright = HandTracker(min_confidence=cfg["hand"]["min_confidence"], ignore_bottom=cfg["hand"]["ignore_bottom"])  # same frames, brightened
    detectors = Detectors(cfg)
    frontal_min = cfg["play_pause"]["frontal"]
    rows, trail = [], deque()
    ends, total = [], 0.0
    for step in STEPS:
        total += step.seconds
        ends.append(total)
    start = last = time.monotonic()
    fps, last_action, last_action_at = 0.0, "", -1e9
    placed = stopped = False
    try:
        while True:
            frame = camera.read()
            now = time.monotonic()
            t = now - start
            if t >= total:
                break
            if frame is None or (camera.duplicate and now - last < MAX_FRAME_GAP):
                continue
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            i = next(k for k, end in enumerate(ends) if t < end)
            step = STEPS[i]
            ms = int(t * 1000)
            hand = tracker.detect(frame, ms)
            hand_bright = tracker_bright.detect(brighten(frame), ms)
            events = detectors.update(hand, now) if step.name != "pregatire" else []
            for e in events:
                last_action, last_action_at = e, now
            rows.append({
                "t": round(t, 3), "step": step.name, "fps": round(fps, 1),
                "brightness": round(float(cv2.resize(frame, (32, 24)).mean()), 1),
                "hand": int(hand is not None), "hand_brightened": int(hand_bright is not None),
                "pose": detectors.pose if step.name != "pregatire" else pose_of(hand, frontal_min),
                "pose_brightened": pose_of(hand_bright, frontal_min),
                "frontal": round(hand.frontal, 3) if hand else "",
                "gesture": hand.gesture if hand else "", "score": round(hand.score, 3) if hand else "",
                "fingers": hand.fingers_up if hand else "",
                "x": round(hand.center[0], 4) if hand else "", "y": round(hand.center[1], 4) if hand else "",
                "events": "|".join(events),
            })

            if hand:
                trail.append((now, hand.center))
            while trail and (not hand or now - trail[0][0] > TRAIL_SECONDS):
                trail.popleft()
            draw_guides(frame, cfg["hand"]["ignore_bottom"])
            if hand:
                draw_hand(frame, hand, [c for _, c in trail])
            draw_hud(frame, hand, active=False, fps=fps, light=light_problem(frame), progress=detectors.progress,
                     action=last_action, action_age=now - last_action_at,
                     ignored=detectors.ignored, pose=detectors.pose)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 115), (w, h - 72), (40, 40, 40), -1)
            _text(frame, f"Pasul {i + 1}/{len(STEPS)}  ({ends[i] - t:4.1f}s)", (10, h - 97), 0.5, (200, 200, 200))
            _text(frame, step.prompt, (10, h - 79), 0.5, (0, 255, 255))
            cv2.imshow(TITLE, frame)
            if not placed:
                cv2.setWindowProperty(TITLE, cv2.WND_PROP_TOPMOST, 1)  # stays visible while you look at it
                placed = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                stopped = True
                break
    finally:
        tracker.close()
        tracker_bright.close()
        camera.off()
        cv2.destroyAllWindows()

    if not rows:
        print("No camera frames were recorded.")
        return
    report = make_report(rows, cfg, stopped)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"report-{stamp}.txt").write_text(report, encoding="utf-8")
    with open(OUT_DIR / f"frames-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"Saved: {OUT_DIR / f'report-{stamp}.txt'}")


def pct(part, whole) -> str:
    return f"{100 * part / whole:3.0f}%" if whole else "  -"


def frontal_tip(rows, current: float) -> str | None:
    """How much the user's open palm faces the camera in the "palma" step, and whether play's threshold fits."""
    values = sorted(r["frontal"] for r in rows
                    if r["step"] == "palma" and r["frontal"] != "" and r["fingers"] != "" and r["fingers"] >= 4)
    if len(values) < 10:
        return None
    typical = values[len(values) // 2]
    if typical >= current + 0.1:
        return f"Palma ta, din fata, are raportul {typical:.2f}: pragul play_pause.frontal = {current} e potrivit."
    suggested = round(max(0.3, typical - 0.15), 2)
    return (f"Palma ta, din fata, are raportul {typical:.2f}, aproape de pragul {current}: play poate rata. "
            f"Sugerat: play_pause.frontal = {suggested}.")


def make_report(rows, cfg, stopped: bool) -> str:
    lines = ["gestures-spotify: test ghidat", "=" * 72]
    if stopped:
        lines.append("(oprit inainte de final: pasii lipsa nu apar)")
    total_false = 0
    lines.append(f"{'pas':12s}{'fps':>5s}{'lumina':>8s}{'mana':>7s}{'+luminat':>9s}{'pozitie':>9s}  gesturi")
    for step in STEPS:
        rs = [r for r in rows if r["step"] == step.name]
        if not rs:
            continue
        n = len(rs)
        hands = sum(r["hand"] for r in rs)
        hands_b = sum(r["hand_brightened"] for r in rs)
        right_pose = pct(sum(r["pose"] == step.pose for r in rs), n) if step.pose else "  -"
        got = Counter(e for r in rs for e in r["events"].split("|") if e)
        unexpected = {e: c for e, c in got.items() if e not in step.expected}
        if step.name != "pregatire":
            total_false += sum(unexpected.values())
        parts = [f"{e} {got.get(e, 0)}/{want}" for e, want in step.expected.items()]
        parts += [f"FALS {e} x{c}" for e, c in unexpected.items()]
        fps = sum(r["fps"] for r in rs) / n
        light = sum(r["brightness"] for r in rs) / n
        lines.append(f"{step.name:12s}{fps:5.0f}{light:8.0f}{pct(hands, n):>7s}{pct(hands_b, n):>9s}"
                     f"{right_pose:>9s}  {', '.join(parts) or '-'}")

    lines += ["", "Ce inseamna:", "  mana      = cadre in care mana a fost gasita",
              "  +luminat  = la fel, pe imaginea luminata automat",
              "  pozitie   = cadre recunoscute corect (PALMA la pasul palma, PUMN la pasul pumn)",
              "  FALS      = gest declansat fara sa fie cerut", ""]

    tips = []
    all_n = len(rows)
    light = sum(r["brightness"] for r in rows) / all_n
    fps = sum(r["fps"] for r in rows) / all_n
    hands = sum(r["hand"] for r in rows if r["step"] != "pregatire")
    hands_b = sum(r["hand_brightened"] for r in rows if r["step"] != "pregatire")
    if light < COVERED_BRIGHTNESS:
        tips.append(f"Imaginea e aproape neagra ({light:.0f}/255): camera e acoperita sau are clapeta inchisa?")
    elif light < DARK_BRIGHTNESS or fps < 20:
        tips.append(f"Lumina: imaginea are luminozitatea {light:.0f}/255 si {fps:.0f} fps. Aprinde o lumina in fata ta: "
                    "camera va da mai multe cadre si mana va fi gasita mai des.")
    if hands_b > hands * 1.15 and hands_b - hands > 15:
        tips.append(f"Imaginea luminata gaseste mana mai des ({hands_b} fata de {hands} cadre): activeaza "
                    "luminarea automata, camera.brighten = true.")
    tip = frontal_tip(rows, cfg["play_pause"]["frontal"])
    if tip:
        tips.append(tip)
    if total_false:
        what = "un gest a pornit" if total_false == 1 else f"{total_false} gesturi au pornit"
        tips.append(f"{what.capitalize()} fara sa fie cerut: vezi liniile cu FALS de mai sus.")
    lines += ["Recomandari:"] + [f"  - {t}" for t in tips or ["Totul arata bine."]]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
