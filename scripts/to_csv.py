# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Convert a Pushshift .zst archive to CSV for quick inspection or sharing. Currenty, it does not filter by keywords or else."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from typing import Sequence

# Allow direct execution from the repository root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
PUSHSHIFT_PERSONAL = ROOT / "tools" / "PushshiftDumps" / "personal"
if str(PUSHSHIFT_PERSONAL) not in sys.path:
    sys.path.insert(0, str(PUSHSHIFT_PERSONAL))

try:  # pragma: no cover - external helper
    from utils import read_obj_zst  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Unable to import read_obj_zst from PushshiftDumps. Initialise the submodule first.") from exc

# Sensible default field sets so novice users get a usable spreadsheet out of the box.
DEFAULT_SUBMISSION_FIELDS: Sequence[str] = (
    "id",
    "created_iso",
    "subreddit",
    "author",
    "title",
    "selftext",
    "score",
    "url",
    "permalink",
)
DEFAULT_COMMENT_FIELDS: Sequence[str] = (
    "id",
    "created_iso",
    "subreddit",
    "author",
    "body",
    "score",
    "link_id",
    "parent_id",
    "permalink",
)


def coerce_created_iso(value: object) -> str:
    """Convert a Pushshift created_utc value to an ISO 8601 UTC string."""
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return ""
    return dt.datetime.utcfromtimestamp(timestamp).isoformat() + "Z"


def normalize_row(obj: dict, fields: Sequence[str]) -> list[str]:
    """Return a list ordered according to *fields* with friendly string values."""
    row: list[str] = []
    for field in fields:
        if field == "created_iso":
            row.append(coerce_created_iso(obj.get("created_utc")))
            continue
        value = obj.get(field)
        row.append("" if value is None else str(value))
    return row


def guess_is_submission(path: Path) -> bool | None:
    """Infer whether a file is submissions or comments based on its name."""
    name = path.name.lower()
    if "submissions" in name:
        return True
    if "comments" in name:
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Pushshift .zst file to CSV")
    parser.add_argument("input", type=Path, help="Path to the .zst file")
    parser.add_argument("output", type=Path, help="Destination CSV path")
    parser.add_argument("--fields", nargs="*", help="Optional list of fields to include in the CSV header")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to write")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    explicit_fields = tuple(args.fields) if args.fields else None
    inferred_is_submission = guess_is_submission(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        written = 0
        fields: Sequence[str] | None = explicit_fields

        if fields is not None:
            writer.writerow(fields)

        # Iterate through every JSON object in the compressed file.
        for obj in read_obj_zst(str(args.input)):
            if fields is None:
                is_submission = inferred_is_submission
                if is_submission is None:
                    is_submission = bool(obj.get("title") or obj.get("selftext"))
                fields = DEFAULT_SUBMISSION_FIELDS if is_submission else DEFAULT_COMMENT_FIELDS
                writer.writerow(fields)
            writer.writerow(normalize_row(obj, fields))
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
