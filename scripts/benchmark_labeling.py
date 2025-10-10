# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Benchmark wrapper to time the labelling pipeline on a small subset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark run_label_sample.py on a subset")
    parser.add_argument("--parquet_dir", required=True, help="Parquet directory to label")
    parser.add_argument("--limit", type=int, default=500, help="Row cap passed to run_label_sample")
    parser.add_argument("--log_level", default="INFO", help="Log level for run_label_sample")
    parser.add_argument("--out_csv", default="data/benchmark_labels.csv", help="Where to write the temporary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_label_sample.py",
        "--parquet_dir",
        args.parquet_dir,
        "--out_csv",
        str(out_path),
        "--limit",
        str(args.limit),
        "--log_level",
        args.log_level,
    ]

    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start

    df = pl.read_csv(out_path)
    rows = df.height
    rows_per_sec = rows / elapsed if elapsed else float("inf")

    print(f"Benchmark complete: {rows} rows in {elapsed:.2f} seconds ({rows_per_sec:.2f} rows/sec)")
    print(f"Output CSV: {out_path}")


if __name__ == "__main__":
    main()
