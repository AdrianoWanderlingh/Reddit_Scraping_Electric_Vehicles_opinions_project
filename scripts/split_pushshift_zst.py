#!/usr/bin/env python3
"""Split Pushshift .zst archives (single file or entire directory tree) with one CLI.

Examples (PowerShell):
  # Split a single file into chunks under OUTDIR (only if >= 1 GiB)
  python scripts/split_pushshift_zst.py --input "C:\path\to\RS_2025-08_comments.zst" --output "C:\out\dir" --delete_input

  # Split every .zst (>= 1 GiB) under INPUT_DIR, mirroring structure under OUTPUT_DIR
  python scripts/split_pushshift_zst.py --input "C:\path\to\reddit_pushshift_dump_2025" --output "C:\path\to\reddit_pushshift_dump_2025_split" --delete_input

Use --dry_run to see what would be processed without making changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Tuple

import zstandard as zstd

# ---- Project paths (kept from your original) --------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PUSHSHIFT_PERSONAL = ROOT / "tools" / "PushshiftDumps" / "personal"
if str(PUSHSHIFT_PERSONAL) not in sys.path:
    sys.path.insert(0, str(PUSHSHIFT_PERSONAL))

from utils import read_obj_zst  # type: ignore

NEWLINE = b"\n"
# Always-on size threshold (no CLI to change). Files smaller than this are skipped.
SIZE_THRESHOLD_BYTES = 1024 ** 3  # 1 GiB


# ---- Core reading/writing ----------------------------------------------------
def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield objects from *path*, raising if the extension is unsupported."""
    if not path.name.endswith(".zst"):
        raise ValueError(f"Unsupported file extension for splitting: {path}")
    yield from read_obj_zst(str(path))


def open_chunk_writer(base_prefix: str, chunk_index: int, out_dir: Path) -> tuple[zstd.ZstdCompressionWriter, Path]:
    """Return a writable zstd stream and the path it writes to."""
    filename = f"{base_prefix}_chunk{chunk_index:05d}.zst"
    path = out_dir / filename
    compressor = zstd.ZstdCompressor(level=3)
    writer = compressor.stream_writer(path.open("wb"))
    return writer, path


def _encode_line(obj: Any) -> bytes:
    """Defensive NDJSON line encoding without double-encoding."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    return json.dumps(obj).encode("utf-8")


def split_file(
    input_path: Path,
    out_dir: Path,
    records_per_chunk: int = 2_000_000,
    progress_interval: int = 200_000,
) -> int:
    """Split *input_path* into chunked .zst files under *out_dir*.

    Returns the number of **non-empty** output chunks created.
    Lazily opens writers so we never create empty chunks.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base_prefix = input_path.stem

    chunk_index = 0  # count of *created* non-empty chunks
    current_chunk_count = 0
    total_count = 0

    writer: Optional[zstd.ZstdCompressionWriter] = None
    current_path: Optional[Path] = None

    try:
        for obj in iter_records(input_path):
            # Open the first (or next) chunk only when we have a record to write
            if writer is None:
                chunk_index += 1
                writer, current_path = open_chunk_writer(base_prefix, chunk_index, out_dir)
                logging.info("Writing chunk %d to %s", chunk_index, current_path)
                current_chunk_count = 0

            writer.write(_encode_line(obj))
            writer.write(NEWLINE)

            current_chunk_count += 1
            total_count += 1

            if progress_interval and total_count % progress_interval == 0:
                logging.info("Processed %d records from %s", total_count, input_path.name)

            if current_chunk_count >= records_per_chunk:
                writer.close()
                logging.info("Finalized chunk %d (%d records)", chunk_index, current_chunk_count)
                writer = None
                current_path = None
                current_chunk_count = 0
    finally:
        if writer is not None:  # close any open non-empty writer
            writer.close()

    logging.info(
        "Split complete. Total records=%d, chunks_created=%d, output_dir=%s",
        total_count, chunk_index, out_dir,
    )
    return chunk_index


# ---- Directory mode helpers --------------------------------------------------
def iter_zst_files(root: Path) -> Iterator[Path]:
    """Yield all .zst files under *root* (recursive), sorted for determinism)."""
    yield from sorted(root.rglob("*.zst"))


def process_directory(
    input_root: Path,
    output_root: Path,
    records_per_chunk: int,
    progress_interval: int,
    delete_input: bool,
    dry_run: bool,
) -> Tuple[int, int, int, int]:
    """Mirror input_root under output_root and split each .zst sequentially, applying the 1 GiB threshold."""
    files = list(iter_zst_files(input_root))
    total = len(files)
    if total == 0:
        logging.info("No .zst files found under %s", input_root)
        return (0, 0, 0, 0)

    processed = skipped = errors = 0

    for idx, src in enumerate(files, 1):
        if not src.is_file():
            skipped += 1
            logging.info("[skip %d/%d] %s (not a regular file)", idx, total, src)
            continue

        try:
            size = src.stat().st_size
        except OSError as e:
            skipped += 1
            logging.info("[skip %d/%d] %s (stat error: %s)", idx, total, src, e)
            continue

        if size < SIZE_THRESHOLD_BYTES:
            skipped += 1
            logging.info("[skip %d/%d] %s (%.2f GiB < 1.00 GiB threshold)", idx, total, src, size / (1024 ** 3))
            continue

        rel = src.relative_to(input_root)
        out_dir = output_root / rel.parent
        base_prefix = src.stem

        logging.info("[start  %d/%d] %s -> %s (prefix=%s, size=%.2f GiB)",
                     idx, total, src, out_dir, base_prefix, size / (1024 ** 3))

        if dry_run:
            logging.info("[dry-run] would split: %s", src)
            processed += 1  # count as considered
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            chunks_created = split_file(
                input_path=src,
                out_dir=out_dir,
                records_per_chunk=records_per_chunk,
                progress_interval=progress_interval,
            )
            logging.info("[done   %d/%d] %s (chunks_created=%d)",
                         idx, total, src, chunks_created)

            if delete_input and chunks_created > 0:
                try:
                    os.unlink(src)
                    logging.info("Deleted source: %s", src)
                except OSError as e:
                    logging.error("Failed to delete %s: %s", src, e)

            processed += 1

        except Exception as e:
            errors += 1
            logging.exception("[error  %d/%d] Failed on %s: %s", idx, total, src, e)

    logging.info("Completed directory split. processed=%d skipped=%d errors=%d total=%d",
                 processed, skipped, errors, total)
    return (processed, skipped, errors, total)


# ---- Unified CLI -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Pushshift .zst files. If --input is a file (>= 1 GiB), split that file. "
                    "If --input is a directory, recurse and split all .zst files >= 1 GiB under it, "
                    "preserving the folder structure under --output."
    )

    parser.add_argument("--input", required=True,
                        help="Path to a single .zst file OR a directory containing .zst files (recursive)")
    parser.add_argument("--output", required=True,
                        help="Output directory. For a single file, chunks go here. For a directory, "
                             "this acts as the root under which the input tree is mirrored.")
    parser.add_argument("--records_per_chunk", type=int, default=2_000_000,
                        help="Maximum number of JSON objects per output chunk (default: 2,000,000)")
    parser.add_argument("--progress_interval", type=int, default=200_000,
                        help="Log progress every N records processed (default: 200,000)")
    parser.add_argument("--delete_input", action="store_true",
                        help="Delete the original input file(s) after successful split")
    parser.add_argument("--dry_run", action="store_true",
                        help="Log what would be processed, without writing chunks or deleting inputs")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"--input not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Safety guard: avoid placing output *inside* input when deleting originals in directory mode.
    if input_path.is_dir() and args.delete_input:
        try:
            output_path.relative_to(input_path)
            raise ValueError("--output must not be inside --input when using directory mode with --delete_input.")
        except ValueError:
            # .relative_to raised => output not under input -> OK
            pass

    if input_path.is_file():
        if not input_path.name.endswith(".zst"):
            raise ValueError(f"--input file must end with .zst, got: {input_path}")

        try:
            size = input_path.stat().st_size
        except OSError as e:
            raise OSError(f"Could not stat --input: {input_path}: {e}") from e

        if size < SIZE_THRESHOLD_BYTES:
            logging.info("Skipping file < 1 GiB: %s (%.2f GiB)", input_path, size / (1024 ** 3))
            return

        if args.dry_run:
            logging.info("[dry-run] would split: %s (%.2f GiB)", input_path, size / (1024 ** 3))
            return

        # Single-file mode
        chunks_created = split_file(
            input_path=input_path,
            out_dir=output_path,
            records_per_chunk=args.records_per_chunk,
            progress_interval=args.progress_interval,
        )
        if args.delete_input and chunks_created > 0:
            try:
                input_path.unlink()
                logging.info("Deleted source: %s", input_path)
            except OSError as e:
                logging.error("Failed to delete %s: %s", input_path, e)
        logging.info("Finished splitting %s (chunks=%d)", input_path.name, chunks_created)
        return

    # Directory mode
    processed, skipped, errors, total = process_directory(
        input_root=input_path,
        output_root=output_path,
        records_per_chunk=args.records_per_chunk,
        progress_interval=args.progress_interval,
        delete_input=args.delete_input,
        dry_run=args.dry_run,
    )
    logging.info("Summary: processed=%d skipped=%d errors=%d total=%d", processed, skipped, errors, total)


if __name__ == "__main__":
    main()
