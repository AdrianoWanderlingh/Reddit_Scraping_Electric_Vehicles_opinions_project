# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Utility script that streams Pushshift .zst archives and writes EV-related rows to Parquet.

The script wraps Watchful1's Pushshift helpers so analysts can filter Reddit comments or
submissions by subreddit, time range, and EV keywords before saving partitioned Parquet files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import the project package when the script is executed directly.
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import datetime as dt
import logging
import subprocess
from typing import Any, Dict, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from evrepo.filters import CandidateFilter
from evrepo.normalize import normalize
from evrepo.utils import load_yaml, read_subreddit_map

# Locate the vendored PushshiftDumps helper scripts.
PUSHSHIFT_ROOT = ROOT / "tools" / "PushshiftDumps"
PUSHSHIFT_PERSONAL = PUSHSHIFT_ROOT / "personal"
PUSHSHIFT_SCRIPTS = PUSHSHIFT_ROOT / "scripts"
TO_CSV_SCRIPT = PUSHSHIFT_SCRIPTS / "to_csv.py"

if str(PUSHSHIFT_PERSONAL) not in sys.path:
    sys.path.insert(0, str(PUSHSHIFT_PERSONAL))

try:
    from utils import read_obj_zst  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Unable to import read_obj_zst from PushshiftDumps") from exc

LOGGER = logging.getLogger("evrepo.ingest")


def iter_records(in_dir: Path, mode: str) -> Iterator[Dict[str, Any]]:
    """Yield raw Pushshift objects from every .zst file in *in_dir* that matches *mode*."""
    suffix = {
        "comments": "_comments.zst",
        "submissions": "_submissions.zst",
        "both": None,
    }[mode]
    for path in sorted(in_dir.rglob("*.zst")):
        if suffix and not path.name.endswith(suffix):
            continue
        LOGGER.debug("Reading %s", path)
        for obj in read_obj_zst(str(path)):
            obj["__source_path"] = str(path)
            yield obj


def parse_time_bounds(start: str, end: str) -> tuple[int, int]:
    """Convert ISO-style start/end strings into epoch second bounds."""
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end)
    if len(end.strip()) <= 10:
        # Treat a plain date as inclusive by extending to the next day.
        end_dt += dt.timedelta(days=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def within_range(utc: int | float | str | None, start_ts: int, end_ts: int) -> bool:
    """Return True when *utc* falls within the requested time window."""
    if utc is None:
        return False
    if isinstance(utc, str):
        try:
            utc = int(utc)
        except ValueError:
            return False
    return start_ts <= int(utc) < end_ts


def ensure_partitions(out_dir: Path, year: int, month: int, subreddit: str) -> Path:
    """Create and return the Parquet partition directory for the given keys."""
    path = out_dir / f"year={year:04d}" / f"month={month:02d}" / f"subreddit={subreddit}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_export_csv(in_dir: Path, out_dir: Path, mode: str) -> None:
    """Optionally emit raw CSV dumps using Pushshift's to_csv helper for debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {
        "comments": "_comments.zst",
        "submissions": "_submissions.zst",
        "both": None,
    }[mode]
    for zst_file in sorted(in_dir.rglob("*.zst")):
        if suffix and not zst_file.name.endswith(suffix):
            continue
        csv_path = out_dir / (zst_file.stem + ".csv")
        if csv_path.exists():
            LOGGER.info("CSV already exists for %s, skipping", zst_file.name)
            continue
        cmd = [sys.executable, str(TO_CSV_SCRIPT), str(zst_file), str(csv_path)]
        LOGGER.info("Generating CSV via PushshiftDumps: %s", " ".join(str(part) for part in cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entry point for converting Pushshift dumps into filtered Parquet files."""
    parser = argparse.ArgumentParser(description="Ingest Pushshift dumps into Parquet")
    parser.add_argument("--in_dir", required=True, help="Directory containing .zst files")
    parser.add_argument("--out_parquet_dir", default="data/parquet", help="Destination for Parquet output")
    parser.add_argument("--out_csv_dir", default=None, help="Optional debug dump directory for raw CSV")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--mode", default="both", choices=["comments", "submissions", "both"])
    parser.add_argument("--ideology_map", default="config/subreddits.yaml")
    parser.add_argument("--keywords", default="config/keywords.yaml")
    parser.add_argument("--neg_filters", default="config/neg_filters.yaml")
    parser.add_argument('--subreddits', nargs='+', help='Optional list of subreddit names to include')
    parser.add_argument('--max_records', type=int, default=None, help='Optional maximum number of records to process')
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    # Normalise any provided subreddit filters so comparisons remain case-insensitive.
    include_subreddits = {s.lower() for s in args.subreddits} if args.subreddits else None

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    start_ts, end_ts = parse_time_bounds(args.start, args.end)

    # Configuration files drive keyword selection and ideology mapping.
    keywords_cfg = load_yaml(args.keywords)
    neg_filters_cfg = load_yaml(args.neg_filters)
    ideology_by_sub = read_subreddit_map(args.ideology_map)
    candidate_filter = CandidateFilter.from_config(keywords_cfg, neg_filters_cfg)

    out_root = Path(args.out_parquet_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.out_csv_dir:
        maybe_export_csv(in_dir, Path(args.out_csv_dir), args.mode)

    # Define the schema once so Parquet partitions line up across batches.
    schema = pa.schema([
        ("id", pa.string()),
        ("is_submission", pa.bool_()),
        ("created_utc", pa.int64()),
        ("year", pa.int32()),
        ("month", pa.int32()),
        ("subreddit", pa.string()),
        ("ideology_group", pa.string()),
        ("author", pa.string()),
        ("text", pa.string()),
        ("permalink", pa.string()),
        ("score", pa.int32()),
        ("source_path", pa.string()),
    ])

    batch: Dict[str, list[Any]] = {name: [] for name in schema.names}
    batch_size = 10_000

    def flush_batch() -> None:
        """Write whatever is in *batch* to partitioned Parquet files and clear the buffer."""
        if not batch["id"]:
            return
        table = pa.table(batch, schema=schema)
        columns = table.to_pydict()
        keys = list(zip(columns["year"], columns["month"], columns["subreddit"]))
        grouped: Dict[tuple[int, int, str], Dict[str, list[Any]]] = {}
        for idx, key in enumerate(keys):
            grouped.setdefault(key, {name: [] for name in schema.names})
            for name in schema.names:
                grouped[key][name].append(columns[name][idx])
        for (year, month, subreddit), rows in grouped.items():
            if year is None or month is None or subreddit is None:
                continue
            partition_dir = ensure_partitions(out_root, int(year), int(month), subreddit)
            part_path = partition_dir / "part.parquet"
            partition_table = pa.table(rows, schema=schema)
            if part_path.exists():
                existing = pq.read_table(part_path)
                partition_table = pa.concat_tables([existing, partition_table], promote=True)
            pq.write_table(partition_table, part_path)
        for name in schema.names:
            batch[name].clear()

    processed = 0
    for obj in iter_records(in_dir, args.mode):
        utc = obj.get("created_utc")
        if not within_range(utc, start_ts, end_ts):
            continue
        record = normalize(obj, ideology_by_sub)

        if include_subreddits:
            subreddit_name = (record.get("subreddit") or "").lower()
            if subreddit_name not in include_subreddits:
                continue

        if not record.get("text"):
            continue
        if not candidate_filter.is_candidate(record["text"]):
            continue

        year = record.get("year")
        month = record.get("month")
        if year is None or month is None:
            utc_val = record.get("created_utc") or utc
            if isinstance(utc_val, str):
                try:
                    utc_val = int(utc_val)
                except ValueError:
                    utc_val = None
            if isinstance(utc_val, int):
                dt_obj = dt.datetime.utcfromtimestamp(utc_val)
                year = year or dt_obj.year
                month = month or dt_obj.month
                record["year"], record["month"] = year, month
        if year is None or month is None:
            LOGGER.debug("Skipping record without year/month: %s", record.get("id"))
            continue

        record["source_path"] = obj.get("__source_path")

        for name in schema.names:
            value = record.get(name)
            if name == "created_utc" and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    value = None
            batch[name].append(value)

        processed += 1
        if args.max_records and processed >= args.max_records:
            LOGGER.info("Reached max_records=%d; stopping early", args.max_records)
            break
        if processed % batch_size == 0:
            flush_batch()
            LOGGER.info("Processed %d records", processed)

    flush_batch()
    LOGGER.info("Ingest complete. Total processed: %d", processed)


if __name__ == "__main__":
    main()
