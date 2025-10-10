#!/usr/bin/env python
"""Quick test (<=3 rows) for stance + sentiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from evrepo.api import run_sentiment, run_stance_label
from evrepo.paths import DEFAULT_DATA_DIR, ensure_parent

LIMIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test for stance and sentiment")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument(
        "--stance_csv",
        default=str(DEFAULT_DATA_DIR / "quick_test" / "stance_labels.csv"),
    )
    parser.add_argument(
        "--sentiment_csv",
        default=str(DEFAULT_DATA_DIR / "quick_test" / "sentiment_labels.csv"),
    )
    parser.add_argument("--log_level", default="WARNING")
    return parser.parse_args()


def load_category_counts(csv_path: Path) -> dict[str, int]:
    df = pl.read_csv(csv_path)
    counts = df["final_category"].value_counts().to_dicts()
    return {row["final_category"]: int(row["count"]) for row in counts}


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    stance_path = ensure_parent(Path(args.stance_csv))
    sentiment_path = ensure_parent(Path(args.sentiment_csv))

    stance_start = time.perf_counter()
    run_stance_label(
        parquet_path,
        stance_path,
        limit=LIMIT,
        fast_model=True,
        batch_size=16,
        log_level=args.log_level,
    )
    stance_elapsed = time.perf_counter() - stance_start
    stance_counts = load_category_counts(stance_path)

    sentiment_start = time.perf_counter()
    run_sentiment(
        parquet_path,
        sentiment_path,
        limit=LIMIT,
        log_level=args.log_level,
    )
    sentiment_elapsed = time.perf_counter() - sentiment_start

    print(
        f"quick stance: limit={LIMIT} elapsed={stance_elapsed:.2f}s rows/sec={LIMIT / stance_elapsed if stance_elapsed else float('inf'):.2f} "
        f"categories={json.dumps(stance_counts, sort_keys=True)}"
    )
    print(
        f"quick sentiment: limit={LIMIT} elapsed={sentiment_elapsed:.2f}s rows/sec={LIMIT / sentiment_elapsed if sentiment_elapsed else float('inf'):.2f}"
    )

    todo_path = Path("TODO.md")
    if todo_path.exists():
        print(f"todo list recorded in {todo_path.resolve()}")


if __name__ == "__main__":
    main()
