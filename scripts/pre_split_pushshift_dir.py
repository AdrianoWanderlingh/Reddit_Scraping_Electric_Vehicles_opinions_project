"""Pre-split large Pushshift .zst archives in a directory tree before ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from split_pushshift_zst import split_file  # type: ignore


def should_skip(path: Path) -> bool:
    """Return True when *path* appears to already be a chunked output."""
    stem = path.stem
    return "_chunk" in stem or "_split" in stem


def resolve_output_path(input_root: Path, output_root: Path, file_path: Path) -> Path:
    """Mirror the relative path of *file_path* from *input_root* into *output_root*."""
    relative = file_path.relative_to(input_root)
    return output_root / relative.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Split oversized Pushshift .zst files before ingestion.")
    parser.add_argument("--input_root", required=True, help="Root directory containing Pushshift .zst files")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Destination root for split files (can be the same as input to split in-place)",
    )
    parser.add_argument(
        "--size_threshold_gb",
        type=float,
        default=1.0,
        help="Files larger than this threshold (in GiB) will be split (default: 1.0)",
    )
    parser.add_argument(
        "--records_per_chunk",
        type=int,
        default=2_000_000,
        help="Maximum JSON objects per chunk file (default: 2,000,000)",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=250_000,
        help="Log progress every N records (default: 250,000)",
    )
    parser.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep the original file after splitting (default behaviour deletes originals)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only log which files would be split without performing any changes",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    threshold_bytes = int(args.size_threshold_gb * (1024**3))
    files_to_split: list[Path] = []

    for path in input_root.rglob("*.zst"):
        if should_skip(path):
            logging.debug("Skipping %s (appears already chunked)", path)
            continue
        if path.stat().st_size < threshold_bytes:
            logging.debug("Skipping %s (size %.2f GiB below threshold)", path, path.stat().st_size / (1024**3))
            continue
        files_to_split.append(path)

    if not files_to_split:
        logging.info("No files exceeded the %.2f GiB threshold under %s", args.size_threshold_gb, input_root)
        return

    logging.info("Splitting %d files larger than %.2f GiB", len(files_to_split), args.size_threshold_gb)

    delete_original = not args.keep_original

    for path in sorted(files_to_split):
        output_dir = resolve_output_path(input_root, output_root, path)
        logging.info("Processing %s -> %s", path, output_dir)
        if args.dry_run:
            continue
        split_file(
            input_path=path,
            out_dir=output_dir,
            records_per_chunk=args.records_per_chunk,
            progress_interval=args.progress_interval,
        )
        if delete_original:
            logging.info("Deleting original file %s to save space", path)
            path.unlink()

    if args.dry_run:
        logging.info("Dry run complete. No files were modified.")


if __name__ == "__main__":
    main()
