<div align="center">

# ✋ gestures-spotify

Control Spotify with hand gestures from your webcam. Real-time, fully local, no account or API key.

<img src="docs/demo.gif" alt="Demo" width="560">

▶ [Watch the full video (45 s)](docs/demo.mp4)

</div>

## Gestures

| Gesture | Action |
|---|---|
| ✊ Fist, held 1.2 s | Pause |
| ✋ Open palm facing the camera, held 1.2 s | Play |
| 👉 Index finger pointing right, held 1 s | Next track |
| 👈 Index finger pointing left, held 1 s | Previous track |
| ✋↻ Open hand turned clockwise, like a knob | Volume up (5% per 15°) |
| ✋↺ Open hand turned counter-clockwise | Volume down |

## How it works

1. **Camera** – frames from the webcam; repeated frames are dropped, so the measured fps is real.
2. **Hand** – MediaPipe returns 21 landmarks and a palm/fist label. Pointing, rotation angle and how much the
   palm faces the camera are computed from the landmarks.
3. **Gestures** – a sign fires after being held still for its time; the volume is a ratchet knob, so turning
   the hand back doesn't undo the change.
4. **Filters** – the bottom of the image (mouse and keyboard area) is masked, and nothing fires for 1 s after
   an action.

## Getting started

Requires Python 3.11+ and a webcam.

```powershell
git clone https://github.com/radasanuflorin/gestures-spotify
cd gestures-spotify
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

| Command | What it does |
|---|---|
| `python -m handgestures.preview` | Live view of the hand analysis |
| `python -m handgestures.calibrate` | 90 s guided test, prints suggested settings |
| `python -m pytest` | 102 tests on simulated hands, no camera needed |

The hand model (~8 MB) is downloaded on first run. All thresholds are in [`config.toml`](config.toml).

## Project structure

```
handgestures/
├── camera.py      capture, real fps, dark-image detection
├── hand.py        landmarks → fingers, pointing, angle
├── gestures.py    held signs and the volume knob
├── detectors.py   combines detectors, masking, cooldown
├── view.py        draws the analysis on the frame
├── preview.py     live preview
└── calibrate.py   guided test
tests/
```

The Spotify integration (media controls, per-app volume, window detection) lives in a private repo.

## Challenges

| Problem | Fix |
|---|---|
| Dim light drops the webcam to ~10 fps | Holds timed in seconds, not frames; duplicate frames skipped |
| Reaching for the mouse looked like a fist | Bottom of the image masked before detection |
| MediaPipe only knows palm and fist | Pointing and knob angle derived from landmarks |
| Second hand in view confused the tracker | Hand switches detected and reset |
| Swipes were unreliable at low fps | Replaced with held signs and a knob |
| Volume dropped when the hand returned | Ratchet knob: only turning away counts |

## License

[MIT](LICENSE)
