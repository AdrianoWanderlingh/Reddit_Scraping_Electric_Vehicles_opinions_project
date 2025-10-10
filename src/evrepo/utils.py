# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Shared helpers for loading YAML configuration files and mapping subreddit ideology."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# We resolve repository-relative paths so command-line scripts can reference config files
# using short relative strings like "config/subreddits.yaml".
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path for *path*, interpreting relative paths from the repo root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def load_yaml(path: str | Path) -> Any:
    """Load a YAML file and return its contents (defaults to an empty dict when blank)."""
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _normalize_subreddit_name(name: str) -> str:
    """Canonicalize subreddit names for lookups.

    - Strips leading "r/" or "/r/" if present.
    - Lowercases the remainder.
    """
    s = (name or "").strip()
    if s.startswith("/r/"):
        s = s[3:]
    elif s.startswith("r/"):
        s = s[2:]
    return s.lower()


def read_subreddit_map(path: str | Path) -> Dict[str, str]:
    """Flatten an ideology → subreddits mapping into {subreddit_lower: ideology}."""
    data = load_yaml(path)
    mapping: Dict[str, str] = {}
    if not isinstance(data, dict):
        return mapping
    for ideology, subreddits in data.items():
        if subreddits is None:
            continue
        if isinstance(subreddits, dict):
            items = subreddits.values()
        else:
            items = [subreddits]
        for entry in items:
            if entry is None:
                continue
            if isinstance(entry, (list, tuple, set)):
                iterable = entry
            else:
                iterable = [entry]
            for subreddit in iterable:
                if not subreddit:
                    continue
                key = _normalize_subreddit_name(str(subreddit))
                if not key:
                    continue
                mapping[key] = str(ideology)
    return mapping
