from handgestures import config
from handgestures.calibrate import frontal_tip, make_report

FPS = 30


def row(t, step, pose="open", events="", hand=1, bright=1, light=120, fps=30, frontal=0.8, fingers=5):
    return {"t": t, "step": step, "fps": fps, "brightness": light, "hand": hand, "hand_brightened": bright,
            "pose": pose, "pose_brightened": pose, "frontal": frontal if hand else "",
            "fingers": fingers if hand else "", "events": events}


def report(rows):
    return make_report(rows, config.DEFAULTS, stopped=False)


def test_report_counts_expected_gestures():
    rows = [row(i / FPS, "dreapta", pose="point_right", events="next" if i in (30, 90, 150) else "")
            for i in range(180)]
    assert "next 3/3" in report(rows)


def test_report_flags_unwanted_gestures():
    rows = [row(i / FPS, "normal", events="next" if i == 10 else "") for i in range(60)]
    text = report(rows)
    assert "FALS next x1" in text and "Un gest a pornit" in text


def test_report_points_at_the_light():
    assert "Aprinde o lumina" in report([row(i / FPS, "palma", light=20, fps=10) for i in range(60)])


def test_report_suspects_a_covered_camera():
    assert "acoperita" in report([row(i / FPS, "palma", light=8, fps=10, hand=0) for i in range(60)])


def test_report_recommends_brightening_when_it_finds_more_hands():
    rows = [row(i / FPS, "palma", hand=int(i % 2 == 0), bright=1) for i in range(120)]
    assert "camera.brighten = true" in report(rows)


def test_frontal_threshold_fits_a_wide_palm():
    rows = [row(i / FPS, "palma", frontal=0.82) for i in range(60)]
    assert "e potrivit" in frontal_tip(rows, 0.55)


def test_frontal_threshold_lowered_for_a_narrow_palm():
    rows = [row(i / FPS, "palma", frontal=0.6) for i in range(60)]
    assert "play_pause.frontal = 0.45" in frontal_tip(rows, 0.55)
