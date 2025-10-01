from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def load_yaml(path: str | Path) -> Any:
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def read_subreddit_map(path: str | Path) -> Dict[str, str]:
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
                mapping[str(subreddit).lower()] = str(ideology)
    return mapping
