# TODO
- Calibrate thresholds in `config/labeling.yaml` after batching improvements.
- Add unit tests to compare `score_all` vs per-subject scoring (within tolerance).
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
  - Orchestrates labeling  sentiment  analysis. Requires --parquet_dir (Parquet root). Defaults outputs under ~/Documents/Reddit_EV_data_and_outputs/results.
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



Actionable checks before long runs
- Ensure `tools/PushshiftDumps` exists and `personal/utils.py` provides `read_obj_zst`.
- Verify `config/subreddits.yaml` covers your targeted subreddits; missing subs will show `ideology_group=None`.
- Start with a narrow timeframe/subreddit ingest (even if files cover all time) to validate the end‑to‑end pipeline and output locations.





# extra info on data size

PS C:\Users\awand\My Drive\UpWork Projects\Reddit_Scraping_Electric_Vehicles_opinions_project> python scripts/ingest_from_pushshiftdumps.py --in_dir "C:\Users\awand\Documents\reddit_pushshift_dump\subreddits24" --out_parquet_dir "C:\Users\awand\Documents\Reddit_EV_data_and_outputs\parquet_subset" --start 2022-06-01 --end 2024-12-31 --mode both --log_level INFO
2025-10-10 18:12:29,610 INFO Processing 1/54 file=Anarcho_Capitalism_comments.zst subreddit=anarcho_capitalism
2025-10-10 18:16:28,340 INFO Finished subreddit=anarcho_capitalism file=Anarcho_Capitalism_comments.zst: written_rows=222 partitions_written+=25 partitions_skipped+=0
2025-10-10 18:16:28,340 INFO Processing 2/54 file=Anarcho_Capitalism_submissions.zst subreddit=anarcho_capitalism
2025-10-10 18:16:49,537 INFO Finished subreddit=anarcho_capitalism file=Anarcho_Capitalism_submissions.zst: written_rows=4 partitions_written+=4 partitions_skipped+=0
2025-10-10 18:16:49,537 INFO Processing 3/54 file=Ask_Politics_comments.zst subreddit=ask_politics
2025-10-10 18:17:07,278 INFO Finished subreddit=ask_politics file=Ask_Politics_comments.zst: written_rows=2 partitions_written+=2 partitions_skipped+=0
2025-10-10 18:17:07,294 INFO Processing 4/54 file=Ask_Politics_submissions.zst subreddit=ask_politics
2025-10-10 18:17:09,777 INFO Finished subreddit=ask_politics file=Ask_Politics_submissions.zst: written_rows=0 partitions_written+=0 partitions_skipped+=0
2025-10-10 18:17:09,777 INFO Processing 5/54 file=AskALiberal_comments.zst subreddit=askaliberal
2025-10-10 18:23:59,072 INFO Finished subreddit=askaliberal file=AskALiberal_comments.zst: written_rows=503 partitions_written+=31 partitions_skipped+=0
2025-10-10 18:23:59,072 INFO Processing 6/54 file=AskALiberal_submissions.zst subreddit=askaliberal
2025-10-10 18:24:09,214 INFO Finished subreddit=askaliberal file=AskALiberal_submissions.zst: written_rows=15 partitions_written+=10 partitions_skipped+=0
2025-10-10 18:24:09,214 INFO Processing 7/54 file=AskConservatives_comments.zst subreddit=askconservatives
2025-10-10 18:31:04,455 INFO Finished subreddit=askconservatives file=AskConservatives_comments.zst: written_rows=562 partitions_written+=31 partitions_skipped+=0
2025-10-10 18:31:04,455 INFO Processing 8/54 file=AskConservatives_submissions.zst subreddit=askconservatives
2025-10-10 18:31:17,966 INFO Finished subreddit=askconservatives file=AskConservatives_submissions.zst: written_rows=15 partitions_written+=11 partitions_skipped+=0
2025-10-10 18:31:17,966 INFO Processing 9/54 file=Askpolitics_comments.zst subreddit=askpolitics
2025-10-10 18:34:00,962 INFO Finished subreddit=askpolitics file=Askpolitics_comments.zst: written_rows=261 partitions_written+=4 partitions_skipped+=0
2025-10-10 18:34:00,962 INFO Processing 10/54 file=Askpolitics_submissions.zst subreddit=askpolitics
2025-10-10 18:34:03,151 INFO Finished subreddit=askpolitics file=Askpolitics_submissions.zst: written_rows=2 partitions_written+=2 partitions_skipped+=0
2025-10-10 18:34:03,151 INFO Processing 11/54 file=AskThe_Donald_comments.zst subreddit=askthe_donald
2025-10-10 18:35:11,976 INFO Finished subreddit=askthe_donald file=AskThe_Donald_comments.zst: written_rows=85 partitions_written+=20 partitions_skipped+=0
2025-10-10 18:35:11,976 INFO Processing 12/54 file=AskThe_Donald_submissions.zst subreddit=askthe_donald
2025-10-10 18:35:23,821 INFO Finished subreddit=askthe_donald file=AskThe_Donald_submissions.zst: written_rows=8 partitions_written+=8 partitions_skipped+=0
2025-10-10 18:35:23,821 INFO Processing 13/54 file=climate_comments.zst subreddit=climate
2025-10-10 18:36:13,843 INFO Finished subreddit=climate file=climate_comments.zst: written_rows=882 partitions_written+=31 partitions_skipped+=0
2025-10-10 18:36:13,843 INFO Processing 14/54 file=climate_submissions.zst subreddit=climate
2025-10-10 18:36:22,846 INFO Finished subreddit=climate file=climate_submissions.zst: written_rows=63 partitions_written+=26 partitions_skipped+=0
2025-10-10 18:36:22,846 INFO Processing 15/54 file=climateskeptics_comments.zst subreddit=climateskeptics
2025-10-10 18:37:19,457 INFO Finished subreddit=climateskeptics file=climateskeptics_comments.zst: written_rows=1019 partitions_written+=31 partitions_skipped+=0
2025-10-10 18:37:19,457 INFO Processing 16/54 file=climateskeptics_submissions.zst subreddit=climateskeptics
2025-10-10 18:37:25,297 INFO Finished subreddit=climateskeptics file=climateskeptics_submissions.zst: written_rows=76 partitions_written+=27 partitions_skipped+=0
2025-10-10 18:37:25,297 INFO Processing 17/54 file=Conservative_comments.zst subreddit=conservative
2025-10-10 18:52:28,702 INFO Finished subreddit=conservative file=Conservative_comments.zst: written_rows=2037 partitions_written+=30 partitions_skipped+=0
2025-10-10 18:52:28,702 INFO Processing 18/54 file=Conservative_News_comments.zst subreddit=conservative_news
2025-10-10 18:52:31,825 INFO Finished subreddit=conservative_news file=Conservative_News_comments.zst: written_rows=1 partitions_written+=1 partitions_skipped+=0
2025-10-10 18:52:31,825 INFO Processing 19/54 file=Conservative_News_submissions.zst subreddit=conservative_news
2025-10-10 18:52:35,983 INFO Finished subreddit=conservative_news file=Conservative_News_submissions.zst: written_rows=13 partitions_written+=6 partitions_skipped+=0
2025-10-10 18:52:35,996 INFO Processing 20/54 file=Conservative_submissions.zst subreddit=conservative
2025-10-10 18:54:18,184 INFO Finished subreddit=conservative file=Conservative_submissions.zst: written_rows=86 partitions_written+=25 partitions_skipped+=0
2025-10-10 18:54:18,184 INFO Processing 21/54 file=DemocraticSocialism_comments.zst subreddit=democraticsocialism
2025-10-10 18:54:59,958 INFO Finished subreddit=democraticsocialism file=DemocraticSocialism_comments.zst: written_rows=10 partitions_written+=7 partitions_skipped+=0
2025-10-10 18:54:59,958 INFO Processing 22/54 file=DemocraticSocialism_submissions.zst subreddit=democraticsocialism
2025-10-10 18:55:07,295 INFO Finished subreddit=democraticsocialism file=DemocraticSocialism_submissions.zst: written_rows=1 partitions_written+=1 partitions_skipped+=0
2025-10-10 18:55:07,295 INFO Processing 23/54 file=democrats_comments.zst subreddit=democrats
2025-10-10 18:57:34,696 INFO Finished subreddit=democrats file=democrats_comments.zst: written_rows=117 partitions_written+=24 partitions_skipped+=0
2025-10-10 18:57:34,696 INFO Processing 24/54 file=democrats_submissions.zst subreddit=democrats
2025-10-10 18:57:58,480 INFO Finished subreddit=democrats file=democrats_submissions.zst: written_rows=13 partitions_written+=8 partitions_skipped+=0
2025-10-10 18:57:58,480 INFO Processing 25/54 file=Enough_Sanders_Spam_comments.zst subreddit=enough_sanders_spam
2025-10-10 19:01:54,308 INFO Finished subreddit=enough_sanders_spam file=Enough_Sanders_Spam_comments.zst: written_rows=83 partitions_written+=26 partitions_skipped+=0
2025-10-10 19:01:54,308 INFO Processing 26/54 file=Enough_Sanders_Spam_submissions.zst subreddit=enough_sanders_spam
2025-10-10 19:02:06,205 INFO Finished subreddit=enough_sanders_spam file=Enough_Sanders_Spam_submissions.zst: written_rows=1 partitions_written+=1 partitions_skipped+=0
2025-10-10 19:02:06,221 INFO Processing 27/54 file=environment_comments.zst subreddit=environment
2025-10-10 19:04:21,750 INFO Finished subreddit=environment file=environment_comments.zst: written_rows=1751 partitions_written+=31 partitions_skipped+=0
2025-10-10 19:04:21,750 INFO Processing 28/54 file=environment_submissions.zst subreddit=environment
2025-10-10 19:04:51,168 INFO Finished subreddit=environment file=environment_submissions.zst: written_rows=94 partitions_written+=30 partitions_skipped+=0
2025-10-10 19:04:51,168 INFO Processing 29/54 file=Feminism_comments.zst subreddit=feminism
2025-10-10 19:06:00,027 INFO Finished subreddit=feminism file=Feminism_comments.zst: written_rows=0 partitions_written+=0 partitions_skipped+=0
2025-10-10 19:06:00,027 INFO Processing 30/54 file=Feminism_submissions.zst subreddit=feminism
2025-10-10 19:06:15,296 INFO Finished subreddit=feminism file=Feminism_submissions.zst: written_rows=1 partitions_written+=1 partitions_skipped+=0
2025-10-10 19:06:15,296 INFO Processing 31/54 file=Futurology_comments.zst subreddit=futurology
2025-10-10 19:19:14,079 INFO Finished subreddit=futurology file=Futurology_comments.zst: written_rows=11546 partitions_written+=33 partitions_skipped+=0
2025-10-10 19:19:14,079 INFO Processing 32/54 file=Futurology_submissions.zst subreddit=futurology
2025-10-10 19:19:51,515 INFO Finished subreddit=futurology file=Futurology_submissions.zst: written_rows=187 partitions_written+=31 partitions_skipped+=0
2025-10-10 19:19:51,515 INFO Processing 33/54 file=JordanPeterson_comments.zst subreddit=jordanpeterson
2025-10-10 19:24:59,669 INFO Finished subreddit=jordanpeterson file=JordanPeterson_comments.zst: written_rows=132 partitions_written+=25 partitions_skipped+=0
2025-10-10 19:24:59,669 INFO Processing 34/54 file=JordanPeterson_submissions.zst subreddit=jordanpeterson
2025-10-10 19:25:29,981 INFO Finished subreddit=jordanpeterson file=JordanPeterson_submissions.zst: written_rows=6 partitions_written+=5 partitions_skipped+=0
2025-10-10 19:25:29,981 INFO Processing 35/54 file=LateStageCapitalism_comments.zst subreddit=latestagecapitalism
2025-10-10 19:30:57,000 INFO Finished subreddit=latestagecapitalism file=LateStageCapitalism_comments.zst: written_rows=219 partitions_written+=28 partitions_skipped+=0
2025-10-10 19:30:57,000 INFO Processing 36/54 file=LateStageCapitalism_submissions.zst subreddit=latestagecapitalism
2025-10-10 19:31:26,433 INFO Finished subreddit=latestagecapitalism file=LateStageCapitalism_submissions.zst: written_rows=5 partitions_written+=5 partitions_skipped+=0
2025-10-10 19:31:26,433 INFO Processing 37/54 file=Liberal_comments.zst subreddit=liberal
2025-10-10 19:31:50,165 INFO Finished subreddit=liberal file=Liberal_comments.zst: written_rows=15 partitions_written+=9 partitions_skipped+=0
2025-10-10 19:31:50,165 INFO Processing 38/54 file=Liberal_submissions.zst subreddit=liberal
2025-10-10 19:31:55,163 INFO Finished subreddit=liberal file=Liberal_submissions.zst: written_rows=2 partitions_written+=2 partitions_skipped+=0
2025-10-10 19:31:55,164 INFO Processing 39/54 file=Libertarian_comments.zst subreddit=libertarian
2025-10-10 19:38:16,737 INFO Finished subreddit=libertarian file=Libertarian_comments.zst: written_rows=180 partitions_written+=27 partitions_skipped+=0
2025-10-10 19:38:16,738 INFO Processing 40/54 file=Libertarian_submissions.zst subreddit=libertarian
2025-10-10 19:38:46,307 INFO Finished subreddit=libertarian file=Libertarian_submissions.zst: written_rows=4 partitions_written+=4 partitions_skipped+=0
2025-10-10 19:38:46,308 INFO Processing 41/54 file=LouderWithCrowder_comments.zst subreddit=louderwithcrowder
2025-10-10 19:40:04,242 INFO Finished subreddit=louderwithcrowder file=LouderWithCrowder_comments.zst: written_rows=107 partitions_written+=11 partitions_skipped+=0
2025-10-10 19:40:04,243 INFO Processing 42/54 file=LouderWithCrowder_submissions.zst subreddit=louderwithcrowder
2025-10-10 19:40:12,930 INFO Finished subreddit=louderwithcrowder file=LouderWithCrowder_submissions.zst: written_rows=3 partitions_written+=3 partitions_skipped+=0
2025-10-10 19:40:12,930 INFO Processing 43/54 file=politics_comments.zst subreddit=politics
2025-10-10 21:35:40,947 INFO Finished subreddit=politics file=politics_comments.zst: written_rows=3243 partitions_written+=31 partitions_skipped+=0
2025-10-10 21:35:40,948 INFO Processing 44/54 file=politics_submissions.zst subreddit=politics
2025-10-10 21:39:17,832 INFO Finished subreddit=politics file=politics_submissions.zst: written_rows=52 partitions_written+=23 partitions_skipped+=0
2025-10-10 21:39:17,832 INFO Processing 45/54 file=progressive_comments.zst subreddit=progressive
2025-10-10 21:39:25,557 INFO Finished subreddit=progressive file=progressive_comments.zst: written_rows=2 partitions_written+=1 partitions_skipped+=0
2025-10-10 21:39:25,557 INFO Processing 46/54 file=progressive_submissions.zst subreddit=progressive
2025-10-10 21:39:30,143 INFO Finished subreddit=progressive file=progressive_submissions.zst: written_rows=0 partitions_written+=0 partitions_skipped+=0
2025-10-10 21:39:30,143 INFO Processing 47/54 file=Republican_comments.zst subreddit=republican
2025-10-10 21:40:33,916 INFO Finished subreddit=republican file=Republican_comments.zst: written_rows=95 partitions_written+=21 partitions_skipped+=0
2025-10-10 21:40:33,916 INFO Processing 48/54 file=Republican_submissions.zst subreddit=republican
2025-10-10 21:40:51,192 INFO Finished subreddit=republican file=Republican_submissions.zst: written_rows=8 partitions_written+=6 partitions_skipped+=0
2025-10-10 21:40:51,193 INFO Processing 49/54 file=socialism_comments.zst subreddit=socialism
2025-10-10 21:42:34,494 INFO Finished subreddit=socialism file=socialism_comments.zst: written_rows=16 partitions_written+=9 partitions_skipped+=0
2025-10-10 21:42:34,495 INFO Processing 50/54 file=socialism_submissions.zst subreddit=socialism
2025-10-10 21:42:54,611 INFO Finished subreddit=socialism file=socialism_submissions.zst: written_rows=2 partitions_written+=2 partitions_skipped+=0
2025-10-10 21:42:54,611 INFO Processing 51/54 file=TwoXChromosomes_comments.zst subreddit=twoxchromosomes
2025-10-10 22:01:17,744 INFO Finished subreddit=twoxchromosomes file=TwoXChromosomes_comments.zst: written_rows=34 partitions_written+=18 partitions_skipped+=0
2025-10-10 22:01:17,745 INFO Processing 52/54 file=TwoXChromosomes_submissions.zst subreddit=twoxchromosomes
2025-10-10 22:02:41,070 INFO Finished subreddit=twoxchromosomes file=TwoXChromosomes_submissions.zst: written_rows=0 partitions_written+=0 partitions_skipped+=0
2025-10-10 22:02:41,070 INFO Processing 53/54 file=WhitePeopleTwitter_comments.zst subreddit=whitepeopletwitter
2025-10-10 22:36:06,163 INFO Finished subreddit=whitepeopletwitter file=WhitePeopleTwitter_comments.zst: written_rows=1543 partitions_written+=31 partitions_skipped+=0
2025-10-10 22:36:06,163 INFO Processing 54/54 file=WhitePeopleTwitter_submissions.zst subreddit=whitepeopletwitter
2025-10-10 22:36:50,636 INFO Finished subreddit=whitepeopletwitter file=WhitePeopleTwitter_submissions.zst: written_rows=0 partitions_written+=0 partitions_skipped+=0
2025-10-10 22:36:50,637 INFO Ingest complete. files=54 raw_seen=349188554 kept_rows=25328 partitions_written=779 partitions_skipped=0 rows_dedup_skipped≈0 elapsed=15861.03s
PS C:\Users\awand\My Drive\UpWork Projects\Reddit_Scraping_Electric_Vehicles_opinions_project> 



