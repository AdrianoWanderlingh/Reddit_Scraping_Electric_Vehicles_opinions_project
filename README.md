# Social Media Scraping - Electric Vehicles Stance

Parsing and Classifying Reddit posts and comments from selected liberal and conservative subreddits to analyze discussions about electric vehicles (EVs) in the US. Classifcation into 9 categories (Pro/Against/Neutral – Product/Mandate/Policy) by ideology (liberal vs conservative), running sentiment analysis, generating trends and n-grams, and exporting final dataset and per-category samples as CSVs.

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
5. **Pre-split any `.zst` files larger than ~1 GiB** using the helper in `scripts/pre_split_pushshift_dir.py`
   (details in the next section). This avoids multi-hour ingestion stalls and frees disk space by deleting
   originals after splitting.
6. Outputs default to `"$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs"`.
   You can override paths via CLI flags.
7. Optional pre-flight check (verifies helpers, shows resolved config paths, layout, and `.zst` manifest):

   ```bash
   python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --print_config --log_level INFO
   ```

---

Actionable checks before long runs
- Ensure `tools/PushshiftDumps` exists and `personal/utils.py` provides `read_obj_zst`.
- Verify `config/subreddits.yaml` covers your targeted subreddits; missing subs will show `ideology_group=None`.
- Start with a narrow timeframe/subreddit ingest (even if files cover all time) to validate the end‑to‑end pipeline and output locations.

## Pre-splitting and filtering Large Pushshift Dumps

Before running ingestion, filter out only the subreddits.yaml files, and split any monthly RC/RS archives larger than 1 GiB into manageable chunks. The script `scripts/split_pushshift_zst.py` mirrors the `comments/` and `submissions/` layout, writes numbered chunk files, and (by default) removes each original `.zst` as soon as its chunks are created.

```powershell
$sourceRoot = "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025"
$splitRoot  = "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025_split"

# Split everything above 1 GiB, delete originals, keep only whitelisted subreddits
python scripts\split_pushshift_zst.py `
  --input  $sourceRoot `
  --output $splitRoot `
  --records_per_chunk 8000000 `
  --progress_interval 100000 `
  --delete_input `
  --subreddit_yaml config/subreddits.yaml
```

Tips:

- Use `--dry_run` to list the files that would be processed.
- Pass `--delete_input` to delete the original dumps (without, memory size will double).
- By default, only rows whose `subreddit` matches the whitelist defined in `config/subreddits.yaml` are retained.
  Pass `--no_subreddit_filter` to keep everything.
- The generated files follow the pattern `RC_2025-01_chunk00001.zst`, so `ingest_from_pushshiftdumps.py` recognises
  comments vs. submissions automatically.

---

## Full Pipeline (All Data)

You can run everything in one command or step-by-step.

### One-shot (all steps chained)

> [!IMPORTANT]  
> Expected run time higher than 9h.

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

> [!IMPORTANT]  
> Data will be filtered according to the secified subreddits and according to the specified filtering keywords (`keywords.yaml`). The keywords have been chosen as EV “product” core terms and context terms (e.g., electric vehicle, EV, BEV, PHEV; batteries/charging/range), mandate terms (e.g., mandate/ban/require), and non‑mandate policy terms (e.g., subsidies/tax credits/standards). A text passes when it contains product core phrases and does not match any negative filters.
> Expected run time higher than 5h.


##### Structure A (per-subreddit dumps, e.g. 2024)

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Documents\reddit_pushshift_dump_2024\subreddits24" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --start 2022-06-01 --end 2024-12-31 --mode both --log_level INFO
```

##### Structure B (monthly RC/RS dumps, e.g. 2025)

Point `--in_dir` at the **split** directory produced by `split_pushshift_dir.py`. The ingester automatically detects
the RC_* (comments) and RS_* (submissions) chunk files inside `comments/` and `submissions/`.

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --start 2025-01-01 --end 2025-06-30 --mode both --log_level INFO
```

By default, the script enforces the subreddit whitelist defined in `config/subreddits.yaml`.
To narrow to a subset of subreddits:

```bash
python scripts/ingest_from_pushshiftdumps.py --in_dir "$env:USERPROFILE\Downloads\reddit_pushshift_dump_2025" --out_parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset_2025" --start 2025-01-01 --end 2025-06-30 --mode both --subreddits r/Conservative r/Libertarian --log_level INFO
```

Each Parquet record includes `is_submission` (True for submissions, False for comments).
You can safely re-run ingestion — it appends only new IDs.

---

#### 2) Label stances

Assigns stance using a natural language inference model (default: fast MNLI `prajjwal1/bert-tiny-mnli`), with:
- **3-way softmax** (neutral mass preserved)
- **Contextual calibration** (null-prompt bias subtraction)
- **Template ensembling** (multiple symmetric paraphrases per class)

> [!IMPORTANT]  
> Expected run time 30-90 mins. Use --limit for smoke tests.

**Typical run (fast, calibrated, full templates):**
```bash
python scripts/label_stance.py `
  --parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" `
  --out_csv     "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\stance_labels.csv" `
  --batch_size 32 --log_level INFO
  ```

Optional flags:

`--large_model` — use the larger MNLI model (defaults to small).

`--templates full|lite|none` — control paraphrase ensembling (default: full).

`--no_calibrate` — disable calibration (bias subtraction), non reccomended.

`--use_weak_rules` --rules_mode simple|full — enable weak-cue fusion.

`--limit` — process a quick subset.

`--overwrite` / `--no_resume` — control resumable behavior.

`--verify` — spot-check single-vs-batch MNLI on a few rows (debug logging).

Environment overrides (optional):

`EVREPO_FAST_NLI_MODEL` — replace the default fast model.

`EVREPO_NLI_MODEL` — replace the default large model.

Performance note: full template ensembling (~3× hypothesis pairs) increases runtime.

With `prajjwal1/bert-tiny-mnli` on CPU:

`--templates full`: ~30–90 min for a medium subset.

`--templates lite`: ~2× speedup with minor quality loss.

`--templates none`: fastest; use for smoke tests.

Resumable: existing IDs in the CSV are skipped automatically.
---

#### 3) Score sentiment

Computes both VADER and transformer sentiment per record.

> [!IMPORTANT]  
> Expected run time 1h.

```bash
python scripts/score_sentiment.py --parquet_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\parquet_subset" --out_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\sentiment_labels.csv" --log_level INFO
```

Resumable: existing IDs in the output CSV are skipped.

---

#### 4) Analysis and plots

Generates an output html with plots.

```bash
python scripts/analyze_ev.py --stance_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\stance_labels.csv" --sentiment_csv "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\sentiment_labels.csv" --out_dir "$env:USERPROFILE\Documents\Reddit_EV_data_and_outputs\results\analysis_subset" --timeframe monthly
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
Inside your `results/` folder (stance/sentiment CSVs and plots) and your Parquet output folder (`parquet_subset`) for ingested data.

**How are subreddits enforced?**
All ingestion runs automatically use the whitelist defined in `config/subreddits.yaml`.
Passing `--subreddits` further narrows the set (intersection).

---

## Citation

If you use this software in your research, please use the GitHub **“Cite this repository”** feature to generate the proper reference.

---

© 2025 OpenFis — Licensed under the MIT License (see `LICENSE`).
