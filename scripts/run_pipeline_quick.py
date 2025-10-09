# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Quick pipeline: stance then sentiment (hard-capped to 3 rows)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

LIMIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick pipeline runner with limit<=3")
    parser.add_argument("--parquet_dir", required=True, help="Parquet directory to sample")
    parser.add_argument("--stance_csv", default="data/stance_labels.csv")
    parser.add_argument("--sentiment_csv", default="data/sentiment_labels.csv")
    parser.add_argument("--merged_csv", default="data/pipeline_merged.csv")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Optional limit (capped at 3)")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--merge", action="store_true", help="Merge stance + sentiment by id")
    return parser.parse_args()


def clamp_limit(value: int | None) -> int:
    if value is None:
        return LIMIT
    return min(value, LIMIT)


def run_cmd(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def merge_outputs(stance_csv: Path, sentiment_csv: Path, out_csv: Path) -> None:
    stance_df = pl.read_csv(stance_csv)
    sentiment_df = pl.read_csv(sentiment_csv)
    sentiment_df = sentiment_df.rename({
        column: f"{column}_sent" if column not in {"id", "created_utc", "subreddit", "ideology_group"} else column
        for column in sentiment_df.columns
    })
    merged = stance_df.join(sentiment_df, on="id", how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(out_csv)


def main() -> None:
    args = parse_args()
    limit = clamp_limit(args.limit)
    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(parquet_dir)

    stance_cmd = [
        sys.executable,
        "scripts/run_label_sample.py",
        "--parquet_dir",
        str(parquet_dir),
        "--out_csv",
        args.stance_csv,
        "--limit",
        str(limit),
        "--log_level",
        args.log_level,
    ]
    sentiment_cmd = [
        sys.executable,
        "scripts/run_sentiment_sample.py",
        "--parquet_dir",
        str(parquet_dir),
        "--out_csv",
        args.sentiment_csv,
        "--limit",
        str(limit),
        "--log_level",
        args.log_level,
    ]

    start = time.perf_counter()
    stance_elapsed = run_cmd(stance_cmd)
    sentiment_elapsed = run_cmd(sentiment_cmd)

    merged_path = "-"
    if args.merge:
        merged_csv = Path(args.merged_csv)
        merge_outputs(Path(args.stance_csv), Path(args.sentiment_csv), merged_csv)
        merged_path = str(merged_csv)

    total_elapsed = time.perf_counter() - start
    print(
        f"pipeline: limit={limit} stance_csv={args.stance_csv} sentiment_csv={args.sentiment_csv} merged_csv={merged_path} elapsed={total_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
