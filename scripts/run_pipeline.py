#!/usr/bin/env python
"""Full pipeline: stance labelling -> sentiment -> analysis."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from evrepo.api import run_sentiment, run_stance_label
from evrepo.paths import DEFAULT_DATA_DIR, ensure_parent
from run_analysis import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full EV stance pipeline")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--out_dir", default=str(DEFAULT_DATA_DIR / "results"))
    parser.add_argument("--stance_csv", default=str(DEFAULT_DATA_DIR / "results" / "stance_labels.csv"))
    parser.add_argument("--sentiment_csv", default=str(DEFAULT_DATA_DIR / "results" / "sentiment_labels.csv"))
    parser.add_argument("--timeframe", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry runs")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--large_model", action="store_true", help="Use the larger MNLI model (default: fast tiny model)")
    parser.add_argument("--use_weak_rules", action="store_true")
    parser.add_argument("--rules_mode", choices=["simple", "full"], default="simple")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite_stance", action="store_true", help="Overwrite stance CSV instead of resuming")
    parser.add_argument("--overwrite_sentiment", action="store_true", help="Overwrite sentiment CSV instead of resuming")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip existing IDs if outputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    parquet_path = Path(args.parquet_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stance_csv = ensure_parent(Path(args.stance_csv))
    sentiment_csv = ensure_parent(Path(args.sentiment_csv))

    stance_start = time.perf_counter()
    stance_stats = run_stance_label(
        parquet_path,
        stance_csv,
        limit=args.limit,
        log_level=args.log_level,
        fast_model=(not args.large_model),
        use_weak_rules=args.use_weak_rules,
        rules_mode=args.rules_mode,
        batch_size=args.batch_size,
        resume=(not args.no_resume),
        overwrite=args.overwrite_stance,
    )
    stance_elapsed = time.perf_counter() - stance_start
    logging.info(
        "stance phase: rows=%d elapsed=%.2fs rows/sec=%.2f",
        stance_stats["rows"],
        stance_stats["elapsed"],
        stance_stats["rows_per_sec"],
    )

    sentiment_start = time.perf_counter()
    sentiment_stats = run_sentiment(
        parquet_path,
        sentiment_csv,
        limit=args.limit,
        log_level=args.log_level,
        resume=(not args.no_resume),
        overwrite=args.overwrite_sentiment,
    )
    sentiment_elapsed = time.perf_counter() - sentiment_start
    logging.info(
        "sentiment phase: rows=%d elapsed=%.2fs rows/sec=%.2f",
        sentiment_stats["rows"],
        sentiment_stats["elapsed"],
        sentiment_stats["rows_per_sec"],
    )

    analysis_start = time.perf_counter()
    analysis_dir = out_dir / "analysis"
    run_analysis(stance_csv, sentiment_csv, analysis_dir, timeframe=args.timeframe, limit=args.limit)
    analysis_elapsed = time.perf_counter() - analysis_start
    logging.info("analysis phase: elapsed=%.2fs", analysis_elapsed)

    total_elapsed = stance_elapsed + sentiment_elapsed + analysis_elapsed
    logging.info(
        "pipeline finished: total_elapsed=%.2fs stance_csv=%s sentiment_csv=%s results_dir=%s",
        total_elapsed,
        stance_csv,
        sentiment_csv,
        analysis_dir,
    )


if __name__ == "__main__":
    main()
