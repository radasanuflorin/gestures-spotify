"""Load config.toml, falling back to defaults for anything missing."""

import copy
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"

DEFAULTS = {
    "camera": {"index": 0, "width": 640, "height": 480, "backend": "dshow", "brighten": False},
    "hand": {"min_confidence": 0.6, "ignore_bottom": 0.25},
    "play_pause": {"hold": 1.2, "max_move": 0.15, "frontal": 0.55},
    "track": {"hold": 1.0, "max_move": 0.15},
    "actions": {"cooldown": 1.0},
    "quit": {"enabled": True, "hold": 1.0},
    "volume": {"step": 5, "step_angle": 15.0, "settle": 0.3},
}

CHOICES = {
    ("camera", "backend"): ("dshow", "msmf"),
}


def load(path: Path = CONFIG_PATH) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        with path.open("rb") as f:
            user = tomllib.load(f)
        for section, values in user.items():
            if section not in cfg:
                raise ValueError(f"{path.name}: unknown section [{section}]")
            for key, value in values.items():
                if key not in cfg[section]:
                    raise ValueError(f"{path.name}: unknown setting {section}.{key}")
                default = cfg[section][key]
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise ValueError(f"{path.name}: {section}.{key} must be a number")
                elif type(value) is not type(default):
                    raise ValueError(f"{path.name}: {section}.{key} must be {type(default).__name__}")
                choices = CHOICES.get((section, key))
                if choices and value not in choices:
                    raise ValueError(f"{path.name}: {section}.{key} must be one of: {', '.join(choices)}")
                cfg[section][key] = value
    return cfg
