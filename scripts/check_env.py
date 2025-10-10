#!/usr/bin/env python
"""Pre-flight checks for the EV stance pipeline.

Validates that Pushshift helpers are available, inspects a dumps directory
to infer available subreddits, compares against your ideology map, and
prints a concise manifest of what would be processed.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Ensure project imports work if run without installation
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evrepo.utils import load_yaml, read_subreddit_map  # type: ignore


def detect_pushshift_helper() -> Tuple[bool, Path]:
    personal = ROOT / "tools" / "PushshiftDumps" / "personal"
    utils_path = personal / "utils.py"
    if str(personal) not in sys.path:
        sys.path.insert(0, str(personal))
    try:  # lazy import to confirm availability
        import utils  # type: ignore  # noqa: F401
        return True, utils_path
    except Exception:
        return utils_path.exists(), utils_path


def enumerate_zst(in_dir: Path) -> List[Path]:
    return sorted(in_dir.rglob("*.zst"))


def infer_file_subreddit(path: Path) -> str | None:
    name = path.stem.lower()
    # expect pattern like '<subreddit>_comments' or '<subreddit>_submissions'
    if "_comments" in name:
        return name.split("_comments", 1)[0]
    if "_submissions" in name:
        return name.split("_submissions", 1)[0]
    # otherwise take the prefix before the first underscore
    return name.split("_", 1)[0] if "_" in name else name


def _normalize_subreddit_name(name: str) -> str:
    s = (name or "").strip()
    s = s.lower()
    if s.startswith("/r/"):
        s = s[3:]
    elif s.startswith("r/"):
        s = s[2:]
    return s


def summarize_files(files: Iterable[Path], include_subs: Set[str] | None) -> Tuple[Counter, Set[str]]:
    kinds = Counter()
    subs: Set[str] = set()
    for p in files:
        name = p.name.lower()
        if include_subs:
            hint = infer_file_subreddit(p)
            if hint not in include_subs:
                continue
        if name.endswith("_comments.zst"):
            kinds["comments"] += 1
        elif name.endswith("_submissions.zst"):
            kinds["submissions"] += 1
        else:
            kinds["unknown"] += 1
        sub = infer_file_subreddit(p)
        if sub:
            subs.add(sub)
    return kinds, subs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight checks for ingestion and labeling")
    parser.add_argument("--in_dir", required=True, help="Directory containing Pushshift .zst dumps")
    parser.add_argument("--ideology_map", default="config/subreddits.yaml")
    parser.add_argument("--keywords", default="config/keywords.yaml")
    parser.add_argument("--neg_filters", default="config/neg_filters.yaml")
    parser.add_argument("--subreddits", nargs="*", help="Optional list of subreddits to focus on")
    parser.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    ok, utils_path = detect_pushshift_helper()
    print(f"Pushshift helper: {'OK' if ok else 'MISSING'} ({utils_path})")

    # Load configuration
    try:
        ideology_map = read_subreddit_map(args.ideology_map)
        keywords = load_yaml(args.keywords)
        neg_filters = load_yaml(args.neg_filters)
    except Exception as exc:
        raise SystemExit(f"Failed to load configuration: {exc}")

    include_subs = {_normalize_subreddit_name(s) for s in args.subreddits} if args.subreddits else None

    files = enumerate_zst(in_dir)
    total_files = len(files)
    kinds, subs_in_files = summarize_files(files, include_subs)

    # Compare subs in files vs ideology config coverage
    config_subs = set(ideology_map.keys())
    considered_subs = subs_in_files if include_subs is None else (subs_in_files & include_subs)
    missing_in_config = sorted(s for s in considered_subs if s not in config_subs)
    # Additional validation: when a filter is provided, also report requested subs missing
    # from config and/or from the files regardless of detection intersection.
    requested_missing_in_config = []
    requested_missing_in_files = []
    if include_subs is not None:
        requested_missing_in_config = sorted([s for s in include_subs if s not in config_subs])
        requested_missing_in_files = sorted([s for s in include_subs if s not in subs_in_files])

    print("--- Manifest ---")
    print(f"Input directory: {in_dir}")
    print(f"Total .zst files found: {total_files} (comments={kinds['comments']}, submissions={kinds['submissions']}, other={kinds['unknown']})")
    if include_subs:
        print(f"Subreddit filter: {sorted(include_subs)}")
    print(f"Detected subreddits in files (sample up to 20): {sorted(list(considered_subs))[:20]}")
    if args.start or args.end:
        print(f"Timeframe: start={args.start or '-'} end={args.end or '-'} (applied per-record during ingestion)")
    print(f"Ideology config: {args.ideology_map} (covered={len(config_subs)} subs)")
    if include_subs:
        if requested_missing_in_files:
            print(f"WARNING: {len(requested_missing_in_files)} requested subreddits have no matching dumps: {requested_missing_in_files[:20]}")
        if requested_missing_in_config:
            print(f"WARNING: {len(requested_missing_in_config)} requested subreddits missing from ideology map: {requested_missing_in_config[:20]}")
        if not requested_missing_in_files and not requested_missing_in_config:
            print("Requested subreddits are present in dumps and covered by ideology map.")
    else:
        if missing_in_config:
            print(f"WARNING: {len(missing_in_config)} subreddits missing from ideology map (first 20 shown): {missing_in_config[:20]}")
        else:
            print("All detected subreddits are present in ideology map.")
    print(f"Keywords config: {args.keywords} (keys={sorted(list((keywords or {}).keys()))[:10]})")
    print(f"Negative filters: {args.neg_filters}")

    # Exit code indicates potential issues
    exit_code = 0
    if not ok:
        exit_code = 2
    print("Pre-flight check completed with exit code:", exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
