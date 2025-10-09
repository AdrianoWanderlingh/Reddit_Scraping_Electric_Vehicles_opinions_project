"""Common path helpers for data/results outputs."""
from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "Documents" / "Reddit_EV_data_and_outputs"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> Path:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
