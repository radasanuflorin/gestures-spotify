"""Webcam capture.

DirectShow is the default. Media Foundation (MSMF) is the alternative; it takes half a minute to open unless
its hardware transforms are off, so they are turned off here (this must happen before OpenCV is loaded).

In dim light a webcam lengthens its exposure and gives fewer frames (often 10 per second instead of 30).
MSMF then repeats the last frame to keep up its nominal rate: repeated frames are flagged as duplicates so
the same image is not analyzed twice and the measured frame rate stays honest.
"""

import logging
import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
DARK_BRIGHTNESS = 45  # mean pixel value (0..255) below which the image counts as too dark
COVERED_BRIGHTNESS = 18  # ...and below which the camera is most likely covered (or its shutter closed)


class CameraError(RuntimeError):
    pass


class Camera:
    """Opens the webcam on the first read, so it can be released while idle and reopened later."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, backend: str = "dshow"):
        self.index, self.width, self.height = index, width, height
        # The chosen backend first, the other one as a fallback.
        self._backends = [backend] + [b for b in BACKENDS if b != backend]
        self._cap = None
        self._last_sample = None
        self.backend = None
        self.duplicate = False  # True when the last read returned the same image as the one before

    @property
    def is_on(self) -> bool:
        return self._cap is not None

    def _open(self):
        for name in self._backends:
            cap = cv2.VideoCapture(self.index, BACKENDS[name])
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, _ = cap.read()
                if ok:
                    self._cap, self.backend = cap, name
                    logging.info("Camera %d on (%s).", self.index, name.upper())
                    return
            cap.release()
            logging.warning("Camera %d could not be opened with %s.", self.index, name.upper())
        raise CameraError(
            f"Could not open webcam {self.index}. Close other apps that use it, or set [camera] index in config.toml."
        )

    def read(self):
        """The next frame, mirrored (so moving right looks like right), or None if the camera gave none."""
        if self._cap is None:
            self._open()
        ok, frame = self._cap.read()
        if not ok:
            return None
        sample = frame[::16, ::16].copy()  # a sparse grid of pixels is enough to spot a repeat
        self.duplicate = self._last_sample is not None and np.array_equal(sample, self._last_sample)
        self._last_sample = sample
        return cv2.flip(frame, 1)

    def off(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_sample = None
            logging.info("Camera off.")


def brightness(frame) -> float:
    # A small copy is enough for the average and much faster.
    return float(cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA).mean())


def is_dark(frame) -> bool:
    return brightness(frame) < DARK_BRIGHTNESS


def light_problem(frame) -> str | None:
    """"covered" (nearly black image), "dark" (dim room), or None."""
    level = brightness(frame)
    return "covered" if level < COVERED_BRIGHTNESS else "dark" if level < DARK_BRIGHTNESS else None


TARGET_BRIGHTNESS = 110  # mean pixel value brighten() aims for


def brighten(frame):
    """Lift a dark image with a gamma curve (shadows up, highlights kept), so the hand stands out better.

    Images that are bright enough are returned unchanged.
    """
    mean = brightness(frame)
    if mean >= DARK_BRIGHTNESS or mean < 1:
        return frame
    gamma = np.log(TARGET_BRIGHTNESS / 255) / np.log(mean / 255)  # maps the mean to the target
    gamma = float(np.clip(gamma, 0.3, 1.0))
    table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(frame, table)
