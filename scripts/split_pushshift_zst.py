#!/usr/bin/env python3
"""Split Pushshift .zst archives with improved performance."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import zstandard as zstd

# ---- Project paths --------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PUSHSHIFT_PERSONAL = ROOT / "tools" / "PushshiftDumps" / "personal"
if str(PUSHSHIFT_PERSONAL) not in sys.path:
    sys.path.insert(0, str(PUSHSHIFT_PERSONAL))

NEWLINE = b"\n"
OUTPUT_FLUSH_BYTES = 8 * 1024 * 1024  # flush writer every 8 MiB
SIZE_THRESHOLD_BYTES = 1024 ** 2  # 1 GiB

# ---- OPTIMIZED reading with byte buffer ----
def iter_records_fast(path: Path, read_size: int = 2 ** 22) -> Iterator[bytes]:
    """Yield raw JSON line bytes from *path* using a zero-copy buffer."""
    if not path.name.endswith(".zst"):
        raise ValueError(f"Unsupported file extension: {path}")

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with path.open("rb") as fh:
        reader = dctx.stream_reader(fh, read_size=read_size)
        buffer = bytearray()

        while True:
            chunk = reader.read(read_size)
            if not chunk:
                break

            buffer.extend(chunk)
            start = 0
            while True:
                idx = buffer.find(NEWLINE, start)
                if idx == -1:
                    # keep remainder for next chunk
                    if start:
                        del buffer[:start]
                    break
                line = buffer[start:idx]
                if line:
                    yield bytes(line)
                start = idx + 1

        if buffer:
            yield bytes(buffer.rstrip())


def open_chunk_writer(base_prefix: str, chunk_index: int, out_dir: Path, compression_level: int = 3) -> tuple[zstd.ZstdCompressionWriter, Path]:
    """Return a writable zstd stream and the path it writes to.
    
    Uses level 3 by default for compression similar to original Pushshift files.
    """
    filename = f"{base_prefix}_chunk{chunk_index:05d}.zst"
    path = out_dir / filename
    
    # Use compression level 3 to match typical Pushshift compression
    # Add threads for parallel compression (huge speedup)
    compressor = zstd.ZstdCompressor(
        level=compression_level,
        threads=-1  # Use all available CPU cores
    )
    writer = compressor.stream_writer(path.open("wb"), closefd=True)
    return writer, path


def split_file(
    input_path: Path,
    out_dir: Path,
    records_per_chunk: int = 24_000_000,
    progress_interval: int = 8_000_000,
    compression_level: int = 3,
) -> Tuple[int, int, float]:
    """Split *input_path* into chunked .zst files under *out_dir*.

    Returns the number of **non-empty** output chunks created.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base_prefix = input_path.stem

    chunk_index = 0
    current_chunk_count = 0
    total_count = 0

    writer: Optional[zstd.ZstdCompressionWriter] = None
    current_path: Optional[Path] = None
    start_time = time.perf_counter()
    out_buffer = bytearray()

    try:
        for line_bytes in iter_records_fast(input_path):
            # Open the first (or next) chunk only when we have a record to write
            if writer is None:
                chunk_index += 1
                writer, current_path = open_chunk_writer(base_prefix, chunk_index, out_dir, compression_level)
                logging.info("Writing chunk %d to %s", chunk_index, current_path)
                current_chunk_count = 0

            # Write raw bytes directly - no encoding/decoding needed!
            out_buffer.extend(line_bytes)
            out_buffer.append(NEWLINE[0])

            current_chunk_count += 1
            total_count += 1
            if len(out_buffer) >= OUTPUT_FLUSH_BYTES:
                writer.write(out_buffer)
                out_buffer.clear()

            if progress_interval and total_count % progress_interval == 0:
                logging.info("Processed %d records from %s", total_count, input_path.name)

            if current_chunk_count >= records_per_chunk:
                if out_buffer:
                    writer.write(out_buffer)
                    out_buffer.clear()
                writer.close()
                logging.info("Finalized chunk %d (%d records)", chunk_index, current_chunk_count)
                writer = None
                current_path = None
                current_chunk_count = 0
    finally:
        if writer is not None:
            if out_buffer:
                writer.write(out_buffer)
                out_buffer.clear()
            writer.close()
            logging.info("Finalized chunk %d (%d records)", chunk_index, current_chunk_count)

    elapsed = time.perf_counter() - start_time
    logging.info(
        "Split complete. Total records=%d, chunks_created=%d, elapsed=%.2fs, output_dir=%s",
        total_count,
        chunk_index,
        elapsed,
        out_dir,
    )
    if total_count > 0 and elapsed > 0:
        rate = total_count / elapsed
        logging.info("Average throughput: %.2f records/s (%.3f ms/record)", rate, 1000.0 / rate)
    return chunk_index, total_count, elapsed


# ---- Directory mode helpers --------------------------------------------------
def iter_zst_files(root: Path) -> Iterator[Path]:
    """Yield all .zst files under *root* (recursive), sorted for determinism."""
    yield from sorted(root.rglob("*.zst"))


def process_directory(
    input_root: Path,
    output_root: Path,
    records_per_chunk: int,
    progress_interval: int,
    compression_level: int,
    delete_input: bool,
    dry_run: bool,
) -> Tuple[int, int, int, int]:
    """Mirror input_root under output_root and split each .zst sequentially."""
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
            processed += 1
            continue

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            chunks_created, records_processed, elapsed = split_file(
                input_path=src,
                out_dir=out_dir,
                records_per_chunk=records_per_chunk,
                progress_interval=progress_interval,
                compression_level=compression_level,
            )
            logging.info("[done   %d/%d] %s (chunks_created=%d, records=%d, elapsed=%.2fs)",
                         idx, total, src, chunks_created, records_processed, elapsed)

            if delete_input and records_processed > 0:
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
        description="Split Pushshift .zst files with optimized performance."
    )

    parser.add_argument("--input", required=True,
                        help="Path to a single .zst file OR a directory containing .zst files")
    parser.add_argument("--output", required=True,
                        help="Output directory for chunks")
    parser.add_argument("--records_per_chunk", type=int, default=2_000_000,
                        help="Maximum records per chunk (default: 2,000,000)")
    parser.add_argument("--progress_interval", type=int, default=200_000,
                        help="Log progress every N records (default: 200,000)")
    parser.add_argument("--compression_level", type=int, default=3,
                        help="Zstd compression level: 3 is the source dafault")
    parser.add_argument("--delete_input", action="store_true",
                        help="Delete original files after successful split")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview what would be processed without making changes")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"--input not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir() and args.delete_input:
        try:
            output_path.relative_to(input_path)
            raise ValueError("--output must not be inside --input when using --delete_input.")
        except ValueError:
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

        chunks_created, records_processed, elapsed = split_file(
            input_path=input_path,
            out_dir=output_path,
            records_per_chunk=args.records_per_chunk,
            progress_interval=args.progress_interval,
            compression_level=args.compression_level,
        )
        if args.delete_input and records_processed > 0:
            try:
                input_path.unlink()
                logging.info("Deleted source: %s", input_path)
            except OSError as e:
                logging.error("Failed to delete %s: %s", input_path, e)
        logging.info(
            "Finished splitting %s (chunks=%d, records=%d, elapsed=%.2fs)",
            input_path.name,
            chunks_created,
            records_processed,
            elapsed,
        )
        return

    # Directory mode
    processed, skipped, errors, total = process_directory(
        input_root=input_path,
        output_root=output_path,
        records_per_chunk=args.records_per_chunk,
        progress_interval=args.progress_interval,
        compression_level=args.compression_level,
        delete_input=args.delete_input,
        dry_run=args.dry_run,
    )
    logging.info("Summary: processed=%d skipped=%d errors=%d total=%d", processed, skipped, errors, total)


if __name__ == "__main__":
    main()
