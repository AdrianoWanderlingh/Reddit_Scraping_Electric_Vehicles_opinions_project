#!/usr/bin/env python3
"""Split Pushshift .zst archives with optional subreddit filtering."""

from __future__ import annotations

import argparse
import json
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

from concurrent.futures import ProcessPoolExecutor, as_completed

from evrepo.utils import read_subreddit_map  # noqa: E402

NEWLINE = b"\n"
OUTPUT_FLUSH_BYTES = 8 * 1024 * 1024  # flush writer every 8 MiB
SIZE_THRESHOLD_BYTES = 1024 ** 3  # 1 GiB


def load_subreddit_whitelist(yaml_path: Path) -> set[str]:
    """Load subreddit names from YAML, returning strings."""
    mapping = read_subreddit_map(str(yaml_path))
    return {name.strip().lower() for name in mapping.keys() if name}


def _split_worker(
    input_path: str,
    out_dir: str,
    records_per_chunk: int,
    progress_interval: int,
    compression_level: int,
    subreddit_whitelist: Optional[tuple[str, ...]],
) -> Tuple[int, int, float, int, int]:
    """Worker wrapper so multiprocessing can call split_file."""
    whitelist_set = set(subreddit_whitelist) if subreddit_whitelist is not None else None
    return split_file(
        input_path=Path(input_path),
        out_dir=Path(out_dir),
        records_per_chunk=records_per_chunk,
        progress_interval=progress_interval,
        compression_level=compression_level,
        subreddit_whitelist=whitelist_set,
    )


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
    records_per_chunk: int = 8_000_000,
    progress_interval: int = 2_000_000,
    compression_level: int = 3,
    subreddit_whitelist: Optional[set[str]] = None,
) -> Tuple[int, int, float, int, int]:
    """Split *input_path* into chunked .zst files under *out_dir*.

    Returns (chunks_created, records_written, elapsed_seconds, total_seen, skipped_filter).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base_prefix = input_path.stem

    chunk_index = 0
    current_chunk_count = 0
    total_seen = 0
    total_written = 0
    skipped_filter = 0

    writer: Optional[zstd.ZstdCompressionWriter] = None
    current_path: Optional[Path] = None
    start_time = time.perf_counter()
    out_buffer = bytearray()

    try:
        for line_bytes in iter_records_fast(input_path):
            total_seen += 1

            if subreddit_whitelist is not None:
                try:
                    record = json.loads(line_bytes.decode("utf-8"))
                except Exception:
                    skipped_filter += 1
                    continue
                subreddit = (record.get("subreddit") or "").strip().lower()
                if subreddit not in subreddit_whitelist:
                    skipped_filter += 1
                    continue

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
            total_written += 1
            if len(out_buffer) >= OUTPUT_FLUSH_BYTES:
                writer.write(out_buffer)
                out_buffer.clear()

            if progress_interval and total_seen % progress_interval == 0:
                logging.info(
                    "Processed %d records from %s (written=%d)",
                    total_seen,
                    input_path.name,
                    total_written,
                )

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
        "Split complete. seen=%d written=%d skipped_filter=%d chunks=%d elapsed=%.2fs output_dir=%s",
        total_seen,
        total_written,
        skipped_filter,
        chunk_index,
        elapsed,
        out_dir,
    )
    if total_written > 0 and elapsed > 0:
        rate = total_written / elapsed
        logging.info("Average throughput: %.2f records/s (%.3f ms/record)", rate, 1000.0 / rate)
    return chunk_index, total_written, elapsed, total_seen, skipped_filter


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
    subreddit_whitelist: Optional[set[str]],
    workers: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """Mirror input_root under output_root and split each .zst sequentially or in parallel."""
    files = list(iter_zst_files(input_root))
    total = len(files)
    if total == 0:
        logging.info("No .zst files found under %s", input_root)
        return (0, 0, 0, 0, 0, 0, 0)

    processed = skipped = errors = 0
    total_written = total_seen = total_skipped_filter = 0
    tasks: list[tuple[str, str, int, int, int, Optional[tuple[str, ...]]]] = []
    whitelist_tuple: Optional[tuple[str, ...]] = (
        tuple(sorted(subreddit_whitelist)) if subreddit_whitelist else None
    )

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
        logging.info("[start  %d/%d] %s -> %s (size=%.2f GiB)",
                     idx, total, src, out_dir, size / (1024 ** 3))

        if dry_run:
            logging.info("[dry-run] would split: %s", src)
            processed += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append(
            (
                str(src),
                str(out_dir),
                records_per_chunk,
                progress_interval,
                compression_level,
                whitelist_tuple,
            )
        )

    if not tasks:
        return (processed, skipped, errors, total, total_written, total_seen, total_skipped_filter)

    worker_count = max(1, min(workers, len(tasks)))

    if worker_count == 1:
        for params in tasks:
            src_path = Path(params[0])
            try:
                chunks_created, records_written, elapsed, seen, skipped_filter = _split_worker(*params)
                logging.info("[done] %s (chunks=%d, records=%d, elapsed=%.2fs)",
                             src_path, chunks_created, records_written, elapsed)
                total_written += records_written
                total_seen += seen
                total_skipped_filter += skipped_filter
                processed += 1
                if delete_input and records_written > 0:
                    try:
                        os.unlink(src_path)
                        logging.info("Deleted source: %s", src_path)
                    except OSError as exc:
                        logging.error("Failed to delete %s: %s", src_path, exc)
            except Exception as exc:
                errors += 1
                logging.exception("[error] Failed on %s: %s", src_path, exc)
    else:
        logging.info("Using %d parallel workers", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_split_worker, *params): params for params in tasks
            }
            for future in as_completed(future_map):
                params = future_map[future]
                src_path = Path(params[0])
                try:
                    chunks_created, records_written, elapsed, seen, skipped_filter = future.result()
                    logging.info("[done] %s (chunks=%d, records=%d, elapsed=%.2fs)",
                                 src_path, chunks_created, records_written, elapsed)
                    total_written += records_written
                    total_seen += seen
                    total_skipped_filter += skipped_filter
                    processed += 1
                    if delete_input and records_written > 0:
                        try:
                            os.unlink(src_path)
                            logging.info("Deleted source: %s", src_path)
                        except OSError as exc:
                            logging.error("Failed to delete %s: %s", src_path, exc)
                except Exception as exc:
                    errors += 1
                    logging.exception("[error] Failed on %s: %s", src_path, exc)

    logging.info(
        "Completed directory split. processed=%d skipped=%d errors=%d total=%d written=%d seen=%d skipped_filter=%d",
        processed,
        skipped,
        errors,
        total,
        total_written,
        total_seen,
        total_skipped_filter,
    )
    return (processed, skipped, errors, total, total_written, total_seen, total_skipped_filter)


# ---- Unified CLI -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Pushshift .zst files with optimized performance."
    )

    parser.add_argument("--input", required=True,
                        help="Path to a single .zst file OR a directory containing .zst files")
    parser.add_argument("--output", required=True,
                        help="Output directory for chunks")
    parser.add_argument("--records_per_chunk", type=int, default=8_000_000,
                        help="Maximum records per chunk (default: 8,000,000)")
    parser.add_argument("--progress_interval", type=int, default=2_000_000,
                        help="Log progress every N records (default: 2,000,000)")
    parser.add_argument("--compression_level", type=int, default=3,
                        help="Zstd compression level: 3 is the source default")
    parser.add_argument("--delete_input", action="store_true",
                        help="Delete original files after successful split")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview what would be processed without making changes")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--subreddit_yaml", default="config/subreddits.yaml",
                        help="Path to subreddit whitelist YAML (default: config/subreddits.yaml)")
    parser.add_argument("--no_subreddit_filter", action="store_true",
                        help="Disable subreddit filtering and keep all records")
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

    subreddit_whitelist: Optional[set[str]] = None
    if not args.no_subreddit_filter and args.subreddit_yaml:
        yaml_path = Path(args.subreddit_yaml)
        if not yaml_path.is_absolute():
            yaml_path = ROOT / yaml_path
        if not yaml_path.exists():
            raise FileNotFoundError(f"Subreddit YAML not found: {yaml_path}")
        subreddit_whitelist = load_subreddit_whitelist(yaml_path)
        if subreddit_whitelist:
            logging.info("Loaded %d whitelisted subreddits from %s", len(subreddit_whitelist), yaml_path)
        else:
            logging.warning("Whitelist %s produced zero entries; no rows will be written", yaml_path)
    else:
        logging.info("Subreddit filtering disabled; all records will be retained")

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

        (
            chunks_created,
            records_written,
            elapsed,
            seen,
            skipped_filter,
        ) = split_file(
            input_path=input_path,
            out_dir=output_path,
            records_per_chunk=args.records_per_chunk,
            progress_interval=args.progress_interval,
            compression_level=args.compression_level,
            subreddit_whitelist=subreddit_whitelist,
        )
        if args.delete_input and records_written > 0:
            try:
                input_path.unlink()
                logging.info("Deleted source: %s", input_path)
            except OSError as e:
                logging.error("Failed to delete %s: %s", input_path, e)
        logging.info(
            "Finished splitting %s (chunks=%d, written=%d, seen=%d, skipped_filter=%d, elapsed=%.2fs)",
            input_path.name,
            chunks_created,
            records_written,
            seen,
            skipped_filter,
            elapsed,
        )
        return

    # Directory mode
    processed, skipped, errors, total_files, total_written, total_seen, total_skipped_filter = process_directory(
        input_root=input_path,
        output_root=output_path,
        records_per_chunk=args.records_per_chunk,
        progress_interval=args.progress_interval,
        compression_level=args.compression_level,
        delete_input=args.delete_input,
        dry_run=args.dry_run,
        subreddit_whitelist=subreddit_whitelist,
        workers=args.workers,
    )
    logging.info(
        "Summary: processed=%d skipped=%d errors=%d total_files=%d total_written=%d total_seen=%d total_skipped_filter=%d",
        processed,
        skipped,
        errors,
        total_files,
        total_written,
        total_seen,
        total_skipped_filter,
    )


if __name__ == "__main__":
    main()
