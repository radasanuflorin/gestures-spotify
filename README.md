<div align="center">

# ✋ gestures-spotify

**Control Spotify with hand gestures from any webcam.** Real-time recognition with MediaPipe, fully local.

<img src="docs/demo.gif" alt="Demo" width="560">

▶ **[Watch the full demo video](docs/demo.mp4)**

</div>

| Gesture | Action |
|---|---|
| ✊ Fist, held 1.2 s | Pause |
| ✋ Open palm, held 1.2 s | Play |
| 👉 / 👈 Finger pointing right / left | Next / previous track |
| ✋↻ Hand turned like a knob | Volume up / down |

## Run

```powershell
pip install -r requirements.txt
python -m handgestures.preview      # live hand analysis
python -m pytest                    # 102 tests, no camera needed
```

> The Spotify control layer is private; this repo is the gesture-recognition engine.

## Challenges

- **Dim light dropped the camera to 10 fps** → repeated frames are skipped, holds are timed in seconds, not frames.
- **Grabbing the mouse looked like a fist** → the bottom of the image is masked before detection.
- **The model only knows palm and fist** → pointing and knob angle come from the 21 hand landmarks.
- **Swipes were unreliable** → replaced with held signs and a ratchet volume knob.

## License

[MIT](LICENSE)
