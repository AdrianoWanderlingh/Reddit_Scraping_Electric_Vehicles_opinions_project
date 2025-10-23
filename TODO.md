# TODO
 - ingestion may benefit from some speedup in file opening, decompression and processing as implemented in 'scripts/split_pushshift_zst.py'
 - Calibrate thresholds in `config/labeling.yaml` after batching improvements.
 - Provide a setup step to pre-download models and eliminate first-run latency.
 - Idempotency markers: consider writing per-partition completion markers (e.g., `_SUCCESS` with start/end range and counts) to avoid scanning existing IDs on resume.
 - Concurrency: add file-locking or atomic rename semantics to avoid partial writes when running multiple workers.
 - Versioning: stamp outputs with model versions and weak-rule config hashes so re-label/re-sentiment runs can detect stale rows and selectively refresh.
 - Metrics: expand skip counters at each stage (per-partition and per-id) to make resume behavior more auditable across logs and manifests.
 - Ingestion filters: record keyword/neg-filter config hash in Parquet metadata; if changed, warn or trigger re-ingest of affected partitions.
 - Input change detection: compute and cache checksums for `.zst` sources to detect new/changed dumps and only process deltas.

## code-debugging

Context: Local Pushshift dumps at `C:\Users\awand\Documents\reddit_pushshift_dump\subreddits24` are organized as flat files per-subreddit, with filenames like `<Subreddit>_comments.zst` and `<Subreddit>_submissions.zst` (no monthly splits). The ingestion code scans all `.zst` recursively and filters rows by `created_utc`, so timeframe subsets are supported at the record level (not by file selection). Notes below list script-by-script checks and any potential mismatches.

- scripts/ingest_from_pushshiftdumps.py
  - Input discovery: uses `Path(args.in_dir).rglob('*.zst')` → OK for `subreddits24` flat folder and for nested structures.
  - Timeframe: enforced per-object via `within_range(created_utc, start_ts, end_ts)` → Subsetting by date is supported even when files cover long spans.
  - Subreddit filtering: `--subreddits` is optional; comparisons are lowercased and use the record’s `subreddit` field, not filenames → OK (case-insensitive). Ensure provided names match API subreddit names (e.g., `AskALiberal`, `Conservative`).
  - EV filtering: uses `config/keywords.yaml` and `config/neg_filters.yaml` via `CandidateFilter` → ensure configs are initialized and include necessary seeds.
  - Ideology mapping: `config/subreddits.yaml` lowercased keys via `read_subreddit_map` → if a subreddit is missing, `ideology_group` will be `None` (allowed). Consider adding missing subs to config.
  - Output layout: partitioned Parquet under `--out_parquet_dir/year=YYYY/month=MM/subreddit=NAME` → OK. Directory will be created on demand.
  - Resume behavior: current code skips IDs already present per partition and appends only new rows; no global early-exit on existing Parquet.
  - External dep: requires `tools/PushshiftDumps/personal/utils.read_obj_zst` → ensure the submodule is initialized; otherwise import error.
  - Performance caveat: filtering timeframe from very large `.zst` files still streams the whole file; for huge subs like `politics_comments.zst` this is expected but long-running.

- scripts/label_stance.py
  - Input: `--parquet_dir` should point to the Parquet root; code scans `**/*.parquet` with Polars → OK with the partition layout produced by ingestion.
  - Output: `--out_csv` arbitrary path (defaults set by caller). Existing files are merged with new rows and deduplicated by `id` → resume-friendly.
  - Limits: `--limit` caps rows at load time for quick checks; for subset-by-subreddit, slice at ingestion or filter downstream when needed.
  - Flags: `--fast_model`, `--use_weak_rules`, `--rules_mode`, `--batch_size` are wired; `--backend onnx` is currently NotImplemented in API (will raise if used).

- scripts/score_sentiment.py
  - Mirrors labeling: scans Parquet root; writes/merges a CSV with `id` deduplication → resume-friendly. `--limit` supported.

- scripts/run_pipeline.py
  - Orchestrates labeling sentiment analysis. Requires --parquet_dir (Parquet root). Defaults outputs under ~/Documents/Reddit_EV_data_and_outputs/results.
  - Respects idempotency implemented in API: re-runs will skip already-processed ids in stance/sentiment CSVs.
  - Supports --overwrite_stance, --overwrite_sentiment, and --no_resume to control rerun behavior explicitly.

- scripts/run_analysis.py
  - Inputs: requires `--stance_csv` and `--sentiment_csv`. Optional `--limit` caps rows post-load (for debugging/preview only). Generates PNGs and a `distinctive_ngrams.csv`.
  - Timeframe aggregation: `--timeframe monthly|quarterly` works; if timestamps are missing/invalid, those rows are dropped from time-based plots.

- scripts/quick_test.py
  - Hard-caps to 3 rows via internal `LIMIT=3`; relies on an existing Parquet dir (not provided by default). Kept as dev utility; no longer referenced in README.

- scripts/run_pipeline_quick.py, scripts/run_label_sample.py, scripts/run_sentiment_sample.py
  - Deprecated wrappers emitting warnings; functionally redirect to current scripts. Safe to ignore for production; consider removal later.

- scripts/benchmark_labeling.py, scripts/tiny_benchmark.py
  - Dev/benchmark utilities. `benchmark_labeling.py` calls the deprecated `run_label_sample.py` (still works, but deprecated). `tiny_benchmark.py` hard-caps to 3 rows.

- scripts/to_csv.py
  - Standalone converter of a single `.zst` to CSV for inspection (no EV filtering). Infers submission/comment fields. Useful for sanity‑checking raw dumps.

- src/evrepo/api.py
  - Parquet loading: uses Polars `scan_parquet('**/*.parquet', glob=True)` selecting minimal columns → OK with partition layout.
  - Resume logic: if output CSV exists, skips existing IDs and writes deduplicated CSVs for both labeling and sentiment.
  - ONNX backend is marked TODO (raises if selected).

- src/evrepo/normalize.py