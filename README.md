# EV Stance Reddit

Classify Reddit posts/comments into 9 categories (Pro/Against/Neutral – Product/Mandate/Policy) by ideology (liberal vs conservative), run sentiment, generate trends and distinctive n-grams, and export final CSV + per-category samples.

---

## Quick Start

1. Install Python 3.10 or newer (Windows Store build or `pyenv`/official installers).
2. Clone the repository and open a terminal in the project folder.
3. Install in editable mode (makes `evrepo` importable):

   ```bash
   python -m pip install -e .
   ```

   The first run downloads dependencies and language models; later runs are faster.
4. Download Pushshift dumps (`.zst`) into a folder you control, e.g.
   `"$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025"`.
5. Outputs default to `"$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs"`.
   You can override paths via CLI flags.
6. Optional pre-flight check (verifies helpers, shows resolved config paths, layout, and `.zst` manifest):

   ```bash
   python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --print_config --log_level INFO
   ```

---

## Full Pipeline (All Data)

You can run everything in one command or step-by-step.

### One-shot (all steps chained)

```bash
python scripts/run_pipeline.py --parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --out_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results" --timeframe monthly --log_level INFO
```

---

### Step-by-step

#### 1) Ingest (Pushshift → Parquet)

Reads `.zst` archives, filters EV-related texts, and writes partitioned Parquet under
`year=YYYY/month=MM/subreddit=NAME`.
The process is resumable—re-running skips IDs already present.

Before ingestion, run a preflight to confirm that the paths, layout, and subreddit list are correct:

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --print_config --log_level INFO
```

##### Structure A (per-subreddit dumps, e.g. 2024)

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Documents\reddit_pushshift_dump_2024\subreddits24" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --start 2022-06-01 --end 2024-12-31 --mode both --log_level INFO
```

##### Structure B (monthly RC/RS dumps, e.g. 2025)

The ingester automatically detects the RC_* (comments) and RS_* (submissions) files inside `comments/` and `submissions/`.

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset_2025" --start 2025-01-01 --end 2025-06-30 --mode both --log_level INFO
```

By default, the script enforces the subreddit whitelist defined in `config/subreddits.yaml`.
To narrow to a subset of subreddits:

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset_2025" --start 2025-01-01 --end 2025-06-30 --mode both --subreddits r/Conservative r/Libertarian --log_level INFO
```

Each Parquet record includes `is_submission` (True for submissions, False for comments).
You can safely re-run ingestion—it appends only new IDs.

---

#### 2) Label stances

Assigns stance using a natural language inference model (default: fast MNLI `prajjwal1/bert-tiny-mnli`).

```bash
python scripts/label_stance.py --parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --out_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\stance_labels.csv" --log_level INFO
```

Optional flags:

* `--large_model`: use a larger MNLI model.
* `--use_weak_rules --rules_mode simple|full`: enable experimental weak-cue fusion.
* `--limit`: process a quick subset.
* `--overwrite` / `--no_resume`: control resumable behavior.

Resumable: existing IDs in the CSV are skipped automatically.

---

#### 3) Score sentiment

Computes both VADER and transformer sentiment per record.

```bash
python scripts/score_sentiment.py --parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --out_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\sentiment_labels.csv" --log_level INFO
```

Resumable: existing IDs in the output CSV are skipped.

---

#### 4) Analysis and plots

Generates `trend_monthly.png`, `sentiment_boxplot.png`, and `distinctive_ngrams.csv`
plus an `analysis_manifest.json` describing generated outputs.

```bash
python scripts/run_analysis.py --stance_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\stance_labels.csv" --sentiment_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\sentiment_labels.csv" --out_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\analysis_subset" --timeframe monthly
```

---

### Reruns and partial runs

* **Ingestion:** idempotent per partition (`year/month/subreddit`). Re-running safely appends missing IDs and skips existing ones.
* **Labeling and sentiment:** idempotent per `id`; already-seen IDs are skipped.
* **Partial data:** you can ingest/label a subset, then expand later without duplication.
* **Full rebuild:** delete target Parquet/CSV outputs and re-run the relevant steps.

---

## CLI Help and Flags

Each script supports `--help` to list options and defaults:

```bash
python scripts/ingest_from_pushshiftdumps.py --help
python scripts/label_stance.py --help
python scripts/score_sentiment.py --help
python scripts/run_pipeline.py --help
python scripts/run_analysis.py --help
```

### Common Flags Overview

#### Ingestion (`ingest_from_pushshiftdumps.py`)

* `--in_dir`: directory containing Pushshift `.zst` files. Both **A** (per-subreddit) and **B** (monthly RC/RS) layouts are detected recursively.
* `--out_parquet_dir`: destination for partitioned Parquet output (default: `data/parquet`).
* `--start` / `--end`: timeframe (e.g., `2025-01-01` to `2025-06-30`). A date `--end` is treated inclusive.
* `--mode`: `comments`, `submissions`, or `both`.
* `--ideology_map`: YAML file containing the enforced subreddit whitelist (default: `config/subreddits.yaml`).
* `--subreddits`: optional subreddit names to **further narrow** the YAML whitelist.
* `--print_config`: log resolved config paths, detected layout, and `.zst` file counts (no ingest).
* `--max_records`: optional limit for testing.
* `--log_level`: logging verbosity.

#### Stance labeling (`label_stance.py`)

* `--parquet_dir`: input Parquet root.
* `--out_csv`: destination CSV for labels.
* `--large_model`: use a larger MNLI model.
* `--use_weak_rules`: incorporate phrase/lexicon weak cues.
* `--rules_mode`: `simple` or `full` weak-rule set.
* `--batch_size`: NLI batch size (adjust for your hardware).
* `--limit`: cap for quick subsets.
* `--overwrite`, `--no_resume`: control overwrite/resume behavior.
* `--log_level`: logging verbosity.

#### Sentiment (`score_sentiment.py`)

* `--parquet_dir`: input Parquet root.
* `--out_csv`: destination CSV for sentiment results.
* `--limit`: quick subset cap.
* `--overwrite`, `--no_resume`: control overwrite/resume behavior.
* `--log_level`: logging verbosity.

#### Full pipeline (`run_pipeline.py`)

* `--parquet_dir`: input Parquet root.
* `--out_dir`: results directory (stance/sentiment and analysis outputs).
* `--timeframe`: analysis aggregation (`monthly` or `quarterly`).
* Other flags (`--large_model`, `--use_weak_rules`, `--rules_mode`, `--limit`, etc.) are forwarded to underlying steps.

---

## Optional Utilities

* `python scripts/run_vulture_report.py` – run Vulture dead-code analysis (`vulture_report.txt`).
* `tools/PushshiftDumps/scripts/to_csv.py` – convert a single `.zst` dump to CSV for inspection.
* `scripts/run_analysis.py` – rerun visualisations/analytics on existing CSVs.

---

## Repository Map

* `config/` – YAML files defining subreddits, keywords, negative filters, and labelling thresholds.
* `scripts/` – CLI helpers (ingest, stance, sentiment, analysis, pipeline).
* `src/evrepo/` – Library modules for normalization, filtering, stance/sentiment scoring, and utilities.
* `tools/PushshiftDumps/` – Vendored utilities for reading `.zst` archives.

---

## Frequently Asked Questions

**Why does the first run take longer?**
It downloads pretrained language models (hundreds of MB). Later runs use cached copies.

**Where is my output?**
Inside your `results/` folder (stance/sentiment CSVs and plots) and your Parquet output folder (`parquet_subset*`) for ingested data.

**How are subreddits enforced?**
All ingestion runs automatically use the whitelist defined in `config/subreddits.yaml`.
Passing `--subreddits` further narrows the set (intersection).

---

## Citation

If you use this software in your research, please use the GitHub **“Cite this repository”** feature to generate the proper reference.

---

© 2025 OpenFis — Licensed under the MIT License (see `LICENSE`).
