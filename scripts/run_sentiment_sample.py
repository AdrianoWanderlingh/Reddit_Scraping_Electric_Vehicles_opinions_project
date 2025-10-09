# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Compute sentiment scores for a small Parquet subset."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import polars as pl

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evrepo.sentiment import SentimentAnalyzer

LOGGER = logging.getLogger("evrepo.run_sentiment_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sentiment metrics from Parquet")
    parser.add_argument("--parquet_dir", default="data/parquet_sample/year=2024", help="Directory containing Parquet partitions")
    parser.add_argument("--out_csv", default="data/sentiment_labels.csv", help="Destination CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap (applied lazily)")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def load_inputs(parquet_dir: Path, limit: int | None) -> pl.DataFrame:
    scan = pl.scan_parquet(str(parquet_dir / "**" / "*.parquet"), glob=True)
    scan = scan.select(["id", "created_utc", "subreddit", "ideology_group", "text"])
    if limit:
        scan = scan.head(limit)
    return scan.collect()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    start_time = time.perf_counter()
    df = load_inputs(parquet_dir, args.limit)
    if df.height == 0:
        LOGGER.warning("No rows found in %s", parquet_dir)
        elapsed = time.perf_counter() - start_time
        LOGGER.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, 0.0)
        return

    analyzer = SentimentAnalyzer()
    results: List[dict] = []
    debug_counter = 0

    for row in df.iter_rows(named=True):
        text_value = row.get("text") or ""
        sentiment = analyzer.score(text_value)
        if LOGGER.isEnabledFor(logging.DEBUG) and debug_counter < 3:
            LOGGER.debug(
                "Row id=%s sentiment vader=%.4f transformer=%s(%.4f)",
                row.get("id"),
                sentiment.vader_compound,
                sentiment.transformer_label,
                sentiment.transformer_score,
            )
            debug_counter += 1
        results.append(
            {
                "id": row.get("id"),
                "created_utc": row.get("created_utc"),
                "subreddit": row.get("subreddit"),
                "ideology_group": row.get("ideology_group"),
                "text": text_value,
                "sent_vader_compound": sentiment.vader_compound,
                "sent_transformer_label": sentiment.transformer_label,
                "sent_transformer_score": sentiment.transformer_score,
            }
        )

    output = pl.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(out_path)
    elapsed = time.perf_counter() - start_time
    rows = len(results)
    rows_per_sec = rows / elapsed if elapsed else float("inf")
    LOGGER.info("Wrote %d rows to %s", rows, out_path)
    LOGGER.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, rows_per_sec)


if __name__ == "__main__":
    main()
