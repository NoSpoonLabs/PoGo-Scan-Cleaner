"""Default locations and the optional TOML config file."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

APP = "scan-cleaner"
DEFAULT_SUBFOLDER = "_PoGo-Scans"

TEMPLATE = """\
# scan-cleaner config. Every value is optional; command-line options win.

# Folders scanned when the command names none.
folders = []

# move: name of the subfolder created next to each found image.
subfolder = "_PoGo-Scans"

# move: one folder for all found images instead of the subfolder ("" = off).
target = ""

# move with target: true = all files directly in target, false = keep the folder structure.
flat = false

# delete: true = delete for good instead of moving to the recycle bin.
permanent = false

# true = an image without the screenshot marker in its metadata is never a match (iOS sets it on every screenshot).
require_screenshot_marker = true

# Where the model file is kept and where run logs go ("" = default location).
model_dir = ""
log_dir = ""
"""


def data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP


def config_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP / "config.toml"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP / "config.toml"


@dataclass
class Settings:
    folders: list[str] = field(default_factory=list)
    subfolder: str = DEFAULT_SUBFOLDER
    target: str = ""
    flat: bool = False
    permanent: bool = False
    require_screenshot_marker: bool = True
    model_dir: str = ""
    log_dir: str = ""

    def model_path(self) -> Path:
        return Path(self.model_dir).expanduser() if self.model_dir else data_dir() / "models"

    def log_path(self) -> Path:
        return Path(self.log_dir).expanduser() if self.log_dir else data_dir() / "logs"


class ConfigError(Exception):
    pass


def load(path: Path | None) -> Settings:
    """Settings from the config file; a missing default file gives the defaults, a missing named file fails."""
    explicit = path is not None
    path = path or config_path()
    if not path.is_file():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return Settings()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    known = {f.name: f for f in fields(Settings)}
    settings = Settings()
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"{path}: unknown key {key!r}")
        default = getattr(settings, key)
        if not isinstance(value, type(default)) or (key == "folders" and not all(isinstance(v, str) for v in value)):
            raise ConfigError(f"{path}: {key!r} must be of type {type(default).__name__}")
        setattr(settings, key, value)
    return settings
