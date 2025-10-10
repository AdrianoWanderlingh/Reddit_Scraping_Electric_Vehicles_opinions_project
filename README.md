# EV Stance Reddit

Classify Reddit posts/comments into 9 categories (Pro/Against/Neutral – Product/Mandate/Policy) by ideology (liberal vs conservative), run sentiment, generate trends and distinctive n-grams, and export final CSV + per-category samples.

## Quick Start
1. Install Python 3.10 or newer (Windows Store build or `pyenv`/official installers).
2. Clone the repository and open a terminal in the project folder.
3. Install in editable mode (makes `evrepo` importable):
   ```bash
   python -m pip install -e .
   ```
   The first run downloads dependencies and language models; later runs are faster.
4. Download Pushshift dumps (`.zst`) into a folder you control, e.g. `C:/data/pushshift`.
5. Outputs default to `~/Documents/Reddit_EV_data_and_outputs`. You can override paths via CLI flags below.
6. Optional pre-flight check (verifies helpers and shows a manifest):
   ```bash
   python scripts/check_env.py --in_dir "C:/path/to/pushshift_dumps" --start 2024-01-01 --end 2024-02-01 --subreddits AskALiberal Conservative
   ```

## Example Workflow (Small Subset)

Goal: run the pipeline on a small subset by limiting timeframe and subreddits to validate everything end‑to‑end.

1) Ingest Pushshift dumps (extract EV‑related rows to Parquet):
```bash
python scripts/ingest_from_pushshiftdumps.py \
  --in_dir "C:/path/to/pushshift_dumps" \
  --out_parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_subset \
  --start 2024-01-01 --end 2024-01-31 \
  --mode both \
  --subreddits AskALiberal Conservative \
  --max_records 10000 \
  --log_level INFO
```
What it does: reads `.zst` archives, filters EV‑related texts, and writes partitioned Parquet under `parquet_subset/` (year/month/subreddit). Resumable: re‑running will skip IDs already present in existing partitions.

2) Label EV stances (fast model by default; fusion optional):
```bash
python scripts/label_stance.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_subset \
  --out_csv    ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
  --log_level INFO
```
Defaults: uses the fast MNLI model (`prajjwal1/bert-tiny-mnli`) and assigns stance using NLI only (no weak‑rule fusion). Optional flags:
- `--large_model` to use the larger MNLI model.
- `--use_weak_rules --rules_mode simple|full` to enable weak‑cue fusion.
Resumable: if the CSV exists, already‑seen IDs are skipped and the file is deduplicated by `id`.

3) Score sentiment:
```bash
python scripts/score_sentiment.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_subset \
  --out_csv    ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
  --log_level INFO
```
What it does: computes VADER and transformer sentiment. Resumable: skips IDs already in the output CSV; writes a deduplicated file.

4) Analysis and plots (on the same subset):
```bash
python scripts/run_analysis.py \
  --stance_csv    ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
  --sentiment_csv ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
  --out_dir       ~/Documents/Reddit_EV_data_and_outputs/results/analysis_subset \
  --timeframe monthly
```
Outputs: `trend_monthly.png`, `sentiment_boxplot.png`, and `distinctive_ngrams.csv` plus an `analysis_manifest.json` describing what was generated.

Tip: you can also run the one‑shot pipeline on the subset:
```bash
python scripts/run_pipeline.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_subset \
  --out_dir     ~/Documents/Reddit_EV_data_and_outputs/results \
  --timeframe   monthly \
  --log_level   INFO
```

## Full Pipeline (All Data)

Run everything in one command or step‑by‑step.

- One‑shot (assumes you have already ingested full Parquet somewhere):
```bash
python scripts/run_pipeline.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_full \
  --out_dir     ~/Documents/Reddit_EV_data_and_outputs/results \
  --timeframe   monthly \
  --log_level   INFO
```

- Step‑by‑step:
1) Ingest (may take a long time):
```bash
python scripts/ingest_from_pushshiftdumps.py \
  --in_dir "C:/path/to/pushshift_dumps" \
  --out_parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_full \
  --start 2005-01-01 --end 2025-06-30 \
  --mode both \
  --log_level INFO
```
2) Label stances:
```bash
python scripts/label_stance.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_full \
  --out_csv    ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
  --fast_model --use_weak_rules --rules_mode simple \
  --log_level  INFO
```
3) Score sentiment:
```bash
python scripts/score_sentiment.py \
  --parquet_dir ~/Documents/Reddit_EV_data_and_outputs/parquet_full \
  --out_csv    ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
  --log_level  INFO
```
4) Analysis and plots:
```bash
python scripts/run_analysis.py \
  --stance_csv    ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
  --sentiment_csv ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
  --out_dir       ~/Documents/Reddit_EV_data_and_outputs/results/analysis \
  --timeframe     monthly
```

Reruns and partial runs:
- Ingestion is idempotent per partition (year/month/subreddit). Re‑running safely appends only missing IDs and skips existing ones.
- Labeling and sentiment are idempotent per `id`. If the output CSV already contains an `id`, it is skipped; new rows are appended and the output is deduplicated by `id`.
- You can ingest/label a subset first, then rerun on the full data later. Only new rows are processed; your CSVs will grow without duplicates.
- To force a rebuild from scratch, delete the target Parquet/CSV outputs and re‑run the relevant steps.

## CLI Help and Flags

Every script supports `--help` to list options and defaults, for example:
```bash
python scripts/ingest_from_pushshiftdumps.py --help
python scripts/label_stance.py --help
python scripts/score_sentiment.py --help
python scripts/run_pipeline.py --help
python scripts/run_analysis.py --help
```

Common flags overview:
- Ingestion (`ingest_from_pushshiftdumps.py`):
  - `--in_dir`: directory containing Pushshift `.zst` files.
  - `--out_parquet_dir`: destination for partitioned Parquet output.
  - `--start/--end`: timeframe (e.g., `2024-01-01` to `2024-06-30`). A date `--end` is treated inclusive.
  - `--mode`: `comments`, `submissions`, or `both`.
  - `--subreddits`: optional subreddit names to include (filters at ingest time).
  - `--max_records`: optional cap for quick subsets.
  - `--log_level`: logging verbosity.
- Stance labeling (`label_stance.py`):
  - `--parquet_dir`: input Parquet root.
  - `--out_csv`: destination CSV for labels.
  - `--large_model`: use a larger MNLI model (default is fast tiny model).
  - `--use_weak_rules`: incorporate phrase/lexicon weak cues.
  - `--rules_mode`: `simple` or `full` weak‑rule set.
  - `--batch_size`: NLI batch size (tune for your hardware).
  - `--limit`: optional cap for quick subsets.
  - `--overwrite`: overwrite output CSV instead of resuming.
  - `--no_resume`: do not skip existing IDs even if output CSV exists.
  - `--log_level`: logging verbosity.
- Sentiment (`score_sentiment.py`):
  - `--parquet_dir`: input Parquet root.
  - `--out_csv`: destination CSV for stance and sentiment.
  - `--limit`: optional cap for quick subsets.
  - `--overwrite`: overwrite output CSV instead of resuming.
  - `--no_resume`: do not skip existing IDs even if output CSV exists.
  - `--log_level`: logging verbosity.
- Full pipeline (`run_pipeline.py`):
  - `--parquet_dir`: input Parquet root.
  - `--out_dir`: results directory (stance/sentiment and analysis outputs).
  - `--timeframe`: analysis aggregation (`monthly` or `quarterly`).
  - `--large_model`, `--use_weak_rules`, `--rules_mode`, `--batch_size`, `--limit`, `--log_level`: forwarded to underlying steps.
  - `--overwrite_stance`, `--overwrite_sentiment`, `--no_resume`: control rerun/overwrite semantics.

## Optional Utilities
- `python scripts/run_vulture_report.py` - run Vulture dead-code analysis (saves `vulture_report.txt`).
- `scripts/to_csv.py` - convert a single `.zst` dump to CSV for inspection.
- `scripts/run_analysis.py` - rerun visualisations/analytics on existing CSVs.

## Repository Map
- `config/` - YAML files defining subreddits, keywords, negative filters, and labelling thresholds.
- `scripts/` - Command-line helpers (ingest, stance, sentiment, analysis, pipeline).
- `src/evrepo/` - Library modules for normalisation, filtering, subject scoring, weak rules, MNLI fusion, sentiment, etc.
- `tools/PushshiftDumps/` - Vendored utilities for reading `.zst` archives.

## Frequently Asked Questions
- Why do some scripts take a long time the first run? They download pretrained language models (hundreds of MB). Subsequent runs use cached copies.
- Where is my output? Look inside the `results/` folder (stance/sentiment CSVs and plots) and your Parquet output folder for ingested data.

## Citation
If you use this software in your research, please use the GitHub "Cite this repository" feature to generate the correct reference.

---

© 2025 OpenFis - Licensed under the MIT License (see `LICENSE`).
