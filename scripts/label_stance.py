#!/usr/bin/env python
"""CLI entrypoint for stance labelling."""
from __future__ import annotations

import argparse
import logging

from evrepo.api import run_stance_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label EV stances from Parquet data")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_weak_rules", action="store_true")
    parser.add_argument("--rules_mode", choices=["simple", "full"], default="simple")
    parser.add_argument("--fast_model", action="store_true")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    run_stance_label(
        args.parquet_dir,
        args.out_csv,
        limit=args.limit,
        log_level=args.log_level,
        debug=args.debug,
        use_weak_rules=args.use_weak_rules,
        rules_mode=args.rules_mode,
        fast_model=args.fast_model,
        backend=args.backend,
        batch_size=args.batch_size,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
