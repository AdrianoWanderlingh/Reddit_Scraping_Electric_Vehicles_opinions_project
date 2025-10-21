# ingest_from_pushshiftdumps.py
# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).

"""
Utility script that streams Pushshift .zst archives and writes EV-related rows to Parquet.

Now supports two input layouts:
- Structure A: per-subreddit files named like SUBREDDIT_submissions.zst / SUBREDDIT_comments.zst
- Structure B: monthly files named like RS_YYYY-MM.zst / RC_YYYY-MM.zst

Key behaviors:
- Auto-detects layout A vs B (or mixed) from filenames.
- Always limits output to subreddits defined in config/subreddits.yaml by default.
  If --subreddits is provided, it further narrows to the intersection with the YAML list.
- For B (monthly) files, subreddit filtering is done per-record (not by filename).
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
import time
import uuid
from typing import Any, Dict, Iterator, Set, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

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


def iter_records(in_dir: Path, mode: str, include_subreddits: Set[str] | None = None) -> Iterator[Dict[str, Any]]:
    """Deprecated by per-file loop; kept for tests."""
    suffix = {
        "comments": "_comments.zst",
        "submissions": "_submissions.zst",
        "both": None,
    }[mode]
    for path in sorted(in_dir.rglob("*.zst")):
        if suffix and not path.name.endswith(suffix):
            continue
        if include_subreddits:
            name_lower = path.stem.lower()
            subreddit_hint = name_lower.split("_", 1)[0]
            if subreddit_hint not in include_subreddits:
                continue
        for obj in read_obj_zst(str(path)):
            obj["__source_path"] = str(path)
            yield obj


def parse_time_bounds(start: str, end: str) -> tuple[int, int]:
    """Convert ISO-style start/end strings into epoch second bounds."""
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end)
    if len(end.strip()) <= 10:  # Treat a plain date as inclusive by extending to the next day.
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

    # >>> CHANGED: accept both A and B style names
    def _matches_mode(filename: str) -> bool:
        if mode == "both":
            return (
                filename.endswith("_comments.zst")
                or filename.endswith("_submissions.zst")
                or filename.startswith("RC_")
                or filename.startswith("RS_")
            )
        if mode == "comments":
            return filename.endswith("_comments.zst") or filename.startswith("RC_")
        if mode == "submissions":
            return filename.endswith("_submissions.zst") or filename.startswith("RS_")
        return True

    for zst_file in sorted(in_dir.rglob("*.zst")):
        if not _matches_mode(zst_file.name):
            continue
        csv_path = out_dir / (zst_file.stem + ".csv")
        if csv_path.exists():
            LOGGER.info("CSV already exists for %s, skipping", zst_file.name)
            continue
        cmd = [sys.executable, str(TO_CSV_SCRIPT), str(zst_file), str(csv_path)]
        LOGGER.info("Generating CSV via PushshiftDumps: %s", " ".join(str(part) for part in cmd))
        subprocess.run(cmd, check=True)


# >>> CHANGED: helpers to detect layout and file kinds
def detect_layout(in_dir: Path) -> str:
    """Return 'A', 'B', 'mixed', or 'unknown' by scanning filenames."""
    has_a = False
    has_b = False
    for p in in_dir.rglob("*.zst"):
        name = p.name
        if name.endswith("_comments.zst") or name.endswith("_submissions.zst"):
            has_a = True
        if name.startswith("RC_") or name.startswith("RS_"):
            has_b = True
        if has_a and has_b:
            return "mixed"
    if has_a:
        return "A"
    if has_b:
        return "B"
    return "unknown"


def file_kind(path: Path) -> str:
    """Return 'comments', 'submissions', or 'unknown' from filename."""
    n = path.name
    if n.endswith("_comments.zst") or n.startswith("RC_"):
        return "comments"
    if n.endswith("_submissions.zst") or n.startswith("RS_"):
        return "submissions"
    return "unknown"


def is_structure_a_file(path: Path) -> bool:
    n = path.name
    return n.endswith("_comments.zst") or n.endswith("_submissions.zst")


def _norm_sub(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("/r/"):
        s = s[3:]
    elif s.startswith("r/"):
        s = s[2:]
    return s


def build_file_list(in_dir: Path, mode: str, include_subreddits: Set[str]) -> List[Path]:
    """Build a file list covering both A and B styles, with smart prefiltering.

    - For A files we can prefilter by subreddit from filename.
    - For B files (RS_/RC_), we cannot know subreddit until reading records, so we keep them.
    """
    candidates = sorted(in_dir.rglob("*.zst"))

    # Filter by mode first
    if mode != "both":
        candidates = [p for p in candidates if file_kind(p) == mode]
    else:
        candidates = [p for p in candidates if file_kind(p) in {"comments", "submissions"}]

    # Prefilter only A-style files by filename subreddit
    def _infer_file_subreddit(p: Path) -> str:
        stem = p.stem.lower()
        if stem.endswith("_comments"):
            return stem[: -len("_comments")]
        if stem.endswith("_submissions"):
            return stem[: -len("_submissions")]
        return stem

    out: List[Path] = []
    for p in candidates:
        if is_structure_a_file(p):
            # only keep if the filename subreddit is in the whitelist
            if _infer_file_subreddit(p) in include_subreddits:
                out.append(p)
        else:
            # monthly files (B): keep; record-level filter will apply
            out.append(p)

    return sorted(out)


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
    parser.add_argument("--subreddits", nargs="+",
                        help="Optional list of subreddit names to further narrow the YAML whitelist")
    parser.add_argument("--print_config", action="store_true",
                        help="Print resolved config paths, whitelist size, detected layout and file counts, then exit")
    parser.add_argument("--max_records", type=int, default=None, help="Optional maximum number of records to process")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

     # Resolve config paths to absolutes for transparency
    args.ideology_map = str(Path(args.ideology_map).resolve())
    args.keywords     = str(Path(args.keywords).resolve())
    args.neg_filters  = str(Path(args.neg_filters).resolve())

    logging.info("Using ideology_map: %s", args.ideology_map)
    logging.info("Using keywords    : %s", args.keywords)
    logging.info("Using neg_filters : %s", args.neg_filters)

    out_root = Path(args.out_parquet_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    start_ts, end_ts = parse_time_bounds(args.start, args.end)

    # Configuration files drive keyword selection and ideology mapping.
    keywords_cfg = load_yaml(args.keywords)
    neg_filters_cfg = load_yaml(args.neg_filters)
    ideology_by_sub = read_subreddit_map(args.ideology_map)

    # default whitelist from YAML; if --subreddits provided, intersect
    yaml_whitelist = {_norm_sub(s) for s in ideology_by_sub.keys()}

    # Optional CLI narrowing to explicit names
    cli_subs: set[str] = {_norm_sub(s) for s in (args.subreddits or [])}
    
    if cli_subs:
        include_subreddits = yaml_whitelist & cli_subs
        missing = cli_subs - yaml_whitelist
        if missing:
            logging.warning(
                "Some --subreddits are not present in %s and will be ignored: %s",
                args.ideology_map,
                ", ".join(sorted(missing)),
            )
    else:
        include_subreddits = yaml_whitelist

    if not include_subreddits:
        raise SystemExit("No subreddits to include after applying YAML map and CLI filters.")

    logging.info("Enforcing whitelist of %d subreddits (e.g., %s ...)",
                 len(include_subreddits), ", ".join(sorted(list(include_subreddits))[:5]))



    candidate_filter = CandidateFilter.from_config(keywords_cfg, neg_filters_cfg)

    if args.out_csv_dir:
        maybe_export_csv(in_dir, Path(args.out_csv_dir), args.mode)

    # Define the schema once so Parquet partitions line up across batches.
    schema = pa.schema(
        [
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
        ]
    )

    batch: Dict[str, list[Any]] = {name: [] for name in schema.names}
    batch_size = 10_000

    # metrics for visibility (totals across all files)
    total_rows_dedup_skipped = 0
    total_partitions_written = 0
    total_partitions_skipped = 0
    total_rows_written = 0
    total_raw_seen = 0
    total_skipped_subreddit = 0
    total_skipped_candidate = 0

    start_time = time.perf_counter()

    # in-memory cache of existing IDs per partition dir
    existing_ids_cache: Dict[Path, set[str]] = {}

    def _load_existing_ids(partition_dir: Path) -> set[str]:
        """Return IDs already present in a partition directory."""
        cached = existing_ids_cache.get(partition_dir)
        if cached is not None:
            return cached
        if not partition_dir.exists():
            existing_ids_cache[partition_dir] = set()
            return existing_ids_cache[partition_dir]
        try:
            dataset = ds.dataset(str(partition_dir), format="parquet")
            if not dataset.files:
                existing_ids_cache[partition_dir] = set()
                return existing_ids_cache[partition_dir]
            table = dataset.to_table(columns=["id"])
            ids = {str(x) for x in table.column("id").to_pylist() if x is not None}
            existing_ids_cache[partition_dir] = ids
            return ids
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed reading existing IDs in %s: %s", partition_dir, exc)
            existing_ids_cache[partition_dir] = set()
            return existing_ids_cache[partition_dir]

    def flush_batch() -> tuple[int, int, int, int]:
        """Write batch to Parquet partitions and clear buffer.
        Returns (rows_written, partitions_written, partitions_skipped, rows_dedup_skipped).
        """
        if not batch["id"]:
            return (0, 0, 0, 0)

        table = pa.table(batch, schema=schema)
        columns = table.to_pydict()
        keys = list(zip(columns["year"], columns["month"], columns["subreddit"]))

        grouped: Dict[tuple[int, int, str], Dict[str, list[Any]]] = {}
        for idx, key in enumerate(keys):
            grouped.setdefault(key, {name: [] for name in schema.names})
            for name in schema.names:
                grouped[key][name].append(columns[name][idx])

        rows_written = 0
        parts_written = 0
        parts_skipped = 0
        rows_dedup_skipped = 0

        for (year, month, subreddit), rows in grouped.items():
            if year is None or month is None or subreddit is None:
                continue
            partition_dir = ensure_partitions(out_root, int(year), int(month), subreddit)

            # Idempotency: skip rows whose IDs already exist in this partition.
            existing_ids = _load_existing_ids(partition_dir)
            orig_len = len(rows["id"]) if rows.get("id") is not None else 0
            keep_idx = [i for i, _id in enumerate(rows["id"]) if _id not in existing_ids]
            if not keep_idx:
                parts_skipped += 1
                continue

            # Filter each column to only new rows
            for name in schema.names:
                col = rows[name]
                rows[name] = [col[i] for i in keep_idx]

            rows_dedup_skipped += max(orig_len - len(keep_idx), 0)

            filename = f"part-{uuid.uuid4().hex[:8]}.parquet"
            part_path = partition_dir / filename
            partition_table = pa.table(rows, schema=schema)
            pq.write_table(partition_table, part_path)
            parts_written += 1
            rows_written += len(rows["id"]) if rows.get("id") is not None else 0

            # Update cache with newly written ids
            existing_ids.update(rows["id"])  # type: ignore[arg-type]

        for name in schema.names:
            batch[name].clear()

        return rows_written, parts_written, parts_skipped, rows_dedup_skipped

    # detect layout for logging only (file building handles both)
    layout = detect_layout(in_dir)
    logging.info("Detected input layout: %s", layout)

    if args.print_config:
        # Quick, non-destructive visibility for diagnostics
        all_zst = [p for p in in_dir.rglob("*.zst")]
        by_kind = {"comments": 0, "submissions": 0, "unknown": 0}
        for p in all_zst:
            by_kind[file_kind(p)] = by_kind.get(file_kind(p), 0) + 1
        logging.info("Found %d .zst files under --in_dir (%d comments, %d submissions, %d unknown)",
                     len(all_zst), by_kind["comments"], by_kind["submissions"], by_kind["unknown"])
        return

    # Build file list (handles A and B)
    all_files = build_file_list(in_dir, args.mode, include_subreddits)
    total_files = len(all_files)
    if total_files == 0:
        logging.warning("No input .zst files matched the requested mode and subreddit whitelist.")
        return

    processed = 0

    for idx, path in enumerate(all_files, start=1):
        kind = file_kind(path)
        logging.info("Processing %d/%d file=%s type=%s", idx, total_files, path.name, kind)

        file_rows_written = 0
        file_parts_written = 0
        file_parts_skipped = 0

        # Stream objects from this file
        for obj in read_obj_zst(str(path)):
            total_raw_seen += 1
            obj["__source_path"] = str(path)

            utc = obj.get("created_utc")
            if not within_range(utc, start_ts, end_ts):
                continue

            record = normalize(obj, ideology_by_sub)

            # Enforce subreddit filter per-record always (covers B monthly files)
            subreddit_name = _norm_sub(record.get("subreddit") or "")
            if subreddit_name not in include_subreddits:
                total_skipped_subreddit += 1
                continue

            text_val = record.get("text")
            if not text_val:
                continue
            if not candidate_filter.is_candidate(text_val):
                total_skipped_candidate += 1
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
                w, pw, ps, d = flush_batch()
                total_rows_written += w
                total_partitions_written += pw
                total_partitions_skipped += ps
                total_rows_dedup_skipped += d
                file_rows_written += w
                file_parts_written += pw
                file_parts_skipped += ps

        # ensure we flush at the end of this file
        w, pw, ps, d = flush_batch()
        total_rows_written += w
        total_partitions_written += pw
        total_partitions_skipped += ps
        total_rows_dedup_skipped += d
        file_rows_written += w
        file_parts_written += pw
        file_parts_skipped += ps

        logging.info(
            "Finished file=%s: written_rows=%d partitions_written+=%d partitions_skipped+=%d",
            path.name,
            file_rows_written,
            file_parts_written,
            file_parts_skipped,
        )

        if args.max_records and processed >= args.max_records:
            break

    elapsed = time.perf_counter() - start_time
    LOGGER.info(
        "Ingest complete. files=%d raw_seen=%d kept_rows=%d partitions_written=%d "
        "partitions_skipped=%d rows_dedup_skipped≈%d elapsed=%.2fs",
        total_files,
        total_raw_seen,
        total_rows_written,
        total_partitions_written,
        total_partitions_skipped,
        total_rows_dedup_skipped,
        elapsed,
    )
    if total_skipped_subreddit:
        LOGGER.info("Rows skipped by subreddit whitelist: %d", total_skipped_subreddit)
    LOGGER.info("Rows skipped by candidate filter: %d", total_skipped_candidate)


if __name__ == "__main__":
    main()
