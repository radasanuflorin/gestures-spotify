import pytest

from handgestures import config


def test_shipped_config_matches_defaults():
    assert config.load() == config.DEFAULTS


def test_missing_file_gives_defaults(tmp_path):
    assert config.load(tmp_path / "nope.toml") == config.DEFAULTS


def test_partial_file_overrides_only_given_keys(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[volume]\nstep = 5\n")
    cfg = config.load(p)
    assert cfg["volume"]["step"] == 5
    assert cfg["volume"]["step_angle"] == config.DEFAULTS["volume"]["step_angle"]


def test_int_accepted_for_float_setting(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[track]\nhold = 2\n")
    assert config.load(p)["track"]["hold"] == 2


@pytest.mark.parametrize(
    "text",
    ["[nope]\nx = 1\n", "[track]\nspeed = 1\n", '[track]\nhold = "long"\n', "[quit]\nenabled = 1\n",
     '[camera]\nbackend = "vfw"\n'],
)
def test_bad_settings_are_rejected(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    with pytest.raises(ValueError):
        config.load(p)
