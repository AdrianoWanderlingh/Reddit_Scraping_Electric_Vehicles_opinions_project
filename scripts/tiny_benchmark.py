# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Run a tiny stance+sentiment benchmark (hard-capped to 3 rows)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from evrepo.api import run_stance_label, run_sentiment
from evrepo.paths import DEFAULT_DATA_DIR, ensure_parent
from evrepo.api import run_stance_label, run_sentiment
LIMIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny benchmark for stance + sentiment")
    parser.add_argument("--parquet_dir", required=True, help="Parquet directory to sample")
    parser.add_argument("--stance_csv", default=str(DEFAULT_DATA_DIR / "tiny_benchmark" / "stance_labels.csv"))
    parser.add_argument("--sentiment_csv", default=str(DEFAULT_DATA_DIR / "tiny_benchmark" / "sentiment_labels.csv"))
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def run_command(func, *args, **kwargs) -> float:
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def stance_summary(csv_path: Path) -> dict:
    df = pl.read_csv(csv_path)
    counts = df["final_category"].value_counts()
    return {row["final_category"]: int(row["count"]) for row in counts.to_dicts()}


def main() -> None:
    args = parse_args()
    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(parquet_dir)

    stance_path = ensure_parent(Path(args.stance_csv))
    sentiment_path = ensure_parent(Path(args.sentiment_csv))

    stance_elapsed = run_command(
        run_stance_label,
        parquet_dir,
        stance_path,
        limit=LIMIT,
        log_level=args.log_level,
        fast_model=True,
        batch_size=16,
    )
    stance_counts = stance_summary(stance_path)

    sentiment_elapsed = run_command(
        run_sentiment,
        parquet_dir,
        sentiment_path,
        limit=LIMIT,
        log_level=args.log_level,
    )

    print(
        f"stance: limit={LIMIT} elapsed={stance_elapsed:.2f}s rows/sec={LIMIT / stance_elapsed if stance_elapsed else float('inf'):.2f} "
        f"categories={json.dumps(stance_counts, sort_keys=True)}"
    )
    print(
        f"sentiment: limit={LIMIT} elapsed={sentiment_elapsed:.2f}s rows/sec={LIMIT / sentiment_elapsed if sentiment_elapsed else float('inf'):.2f}"
    )


if __name__ == "__main__":
    main()
