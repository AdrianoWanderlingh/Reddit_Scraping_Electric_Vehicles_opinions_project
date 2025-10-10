#!/usr/bin/env python
"""CLI entrypoint for sentiment scoring."""
from __future__ import annotations

import argparse
import logging

from evrepo.api import run_sentiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sentiment metrics from Parquet data")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV instead of resuming")
    parser.add_argument("--no_resume", action="store_true", help="Do not skip existing IDs if output CSV exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    run_sentiment(
        args.parquet_dir,
        args.out_csv,
        limit=args.limit,
        log_level=args.log_level,
        debug=args.debug,
        resume=(not args.no_resume),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
