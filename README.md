<div align="center">

# ✋ gestures-spotify

Control Spotify with hand gestures from your webcam.

<img src="docs/demo.gif" alt="Demo" width="560">

▶ [Watch the full video (45 s)](docs/demo.mp4)

</div>

## Gestures

✊ pause · ✋ play · 👉 next · 👈 previous · ✋↻ volume

## Run

```powershell
pip install -r requirements.txt
python -m handgestures.preview
```

## Challenges

- Low fps in dim light → timing in seconds, not frames
- Mouse grabs looked like gestures → bottom of the image ignored
- Pointing and knob unknown to the model → computed from hand landmarks
- Volume undid itself → ratchet knob

## License

MIT
