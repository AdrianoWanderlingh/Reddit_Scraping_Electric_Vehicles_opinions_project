# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Run a tiny stance+sentiment benchmark (hard-capped to 3 rows)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

LIMIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny benchmark for stance + sentiment")
    parser.add_argument("--parquet_dir", required=True, help="Parquet directory to sample")
    parser.add_argument("--stance_csv", default="data/stance_labels.csv")
    parser.add_argument("--sentiment_csv", default="data/sentiment_labels.csv")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def run_command(cmd: list[str]) -> tuple[float, str]:
    start = time.perf_counter()
    completed = subprocess.run(cmd, check=True, text=True, capture_output=False)
    elapsed = time.perf_counter() - start
    return elapsed, ""


def stance_summary(csv_path: Path) -> dict:
    df = pl.read_csv(csv_path)
    counts = df["final_category"].value_counts()
    return {row["final_category"]: int(row["count"]) for row in counts.to_dicts()}


def main() -> None:
    args = parse_args()
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
        str(LIMIT),
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
        str(LIMIT),
        "--log_level",
        args.log_level,
    ]

    stance_elapsed, _ = run_command(stance_cmd)
    stance_counts = stance_summary(Path(args.stance_csv))

    sentiment_elapsed, _ = run_command(sentiment_cmd)

    print(
        f"stance: limit={LIMIT} elapsed={stance_elapsed:.2f}s rows/sec={LIMIT / stance_elapsed if stance_elapsed else float('inf'):.2f} "
        f"categories={json.dumps(stance_counts, sort_keys=True)}"
    )
    print(
        f"sentiment: limit={LIMIT} elapsed={sentiment_elapsed:.2f}s rows/sec={LIMIT / sentiment_elapsed if sentiment_elapsed else float('inf'):.2f}"
    )


if __name__ == "__main__":
    main()
