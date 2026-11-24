import numpy as np

from handgestures.camera import DARK_BRIGHTNESS, brighten, is_dark, light_problem


def gradient(mean):
    """An image with a range of values around `mean`."""
    row = np.clip(np.linspace(mean * 0.2, mean * 1.8, 640), 0, 255).astype(np.uint8)
    return np.repeat(np.repeat(row[None, :, None], 480, axis=0), 3, axis=2)


def test_dark_image_is_brightened_and_keeps_its_order():
    img = gradient(25)
    out = brighten(img)
    assert is_dark(img) and not is_dark(out)
    assert out.mean() > 3 * img.mean()
    assert np.all(np.diff(out[0, :, 0].astype(int)) >= 0)  # darker stays darker


def test_bright_image_is_unchanged():
    img = gradient(DARK_BRIGHTNESS + 40)
    assert brighten(img) is img


def test_black_image_is_unchanged():
    img = np.zeros((480, 640, 3), np.uint8)
    assert brighten(img) is img


def test_light_problem():
    assert light_problem(gradient(8)) == "covered"
    assert light_problem(gradient(30)) == "dark"
    assert light_problem(gradient(120)) is None
