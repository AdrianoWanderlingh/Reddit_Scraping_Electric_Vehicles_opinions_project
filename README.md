# EV Stance Reddit

Classify Reddit posts/comments into 9 categories (Pro/Against/Neutral × Product/Mandate/Policy) by ideology (liberal vs conservative), run sentiment, generate trends and distinctive n-grams, and export final CSV + per-category samples.

## Quick Start
1. **Install Python 3.10 or newer.** On Windows you can use the Microsoft Store build; on macOS/Linux use the official installer or `pyenv`.
2. **Clone the repository** and open a terminal in the project folder.
3. **Install the project in editable mode** (puts `evrepo` on your `PYTHONPATH` automatically):
   ```bash
   python -m pip install -e .
   ```
   *(The first run downloads dependencies and language models; later runs are faster.)*
4. **Download Pushshift dumps** (the `.zst` files) into a folder you control, e.g. `C:/data/pushshift`.
5. **Outputs are stored outside the repo**: the tools default to `~/Documents/Reddit_EV_data_and_outputs`. Adjust CLI paths if you prefer a different location.
6. **Try the quick test (≤3 rows)** to confirm everything works:
   ```bash
   python scripts/quick_test.py --parquet_dir data/parquet_sample
   # pass --log_level INFO for verbose timings if desired
   ```
   You should see timing summaries for stance and sentiment and two CSVs under `data/`.

## Example Workflow (Ingestion → Labelling → Sentiment → Analysis)

1. **Ingest Pushshift dumps** (extract EV discussions into Parquet):
   ```bash
   python scripts/ingest_from_pushshiftdumps.py \
     --in_dir "C:/path/to/pushshift_dumps" \
     --out_parquet_dir data/parquet_sample \
     --start 2024-01-01 --end 2024-06-30 \
     --mode submissions \
     --subreddits AskALiberal \
     --max_records 2000 \
     --log_level INFO
   ```
   *Reads `.zst` archives, filters EV-related text, and writes partitioned Parquet files under `data/parquet_sample/`.*

2. **Label EV stances** (fast path – subject keywords + batched MNLI):
   ```bash
   python scripts/label_stance.py \
     --parquet_dir data/parquet_sample \
     --out_csv ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
     --log_level INFO
   ```
   *Creates `results/stance_labels.csv` with columns such as `final_subject`, `final_category`, `fused_pro`, `fused_anti`, `confidence`.*

3. **Score sentiment separately**:
   ```bash
   python scripts/score_sentiment.py \
     --parquet_dir data/parquet_sample \
     --out_csv ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
     --log_level INFO
   ```
   *Produces `results/sentiment_labels.csv` containing VADER and transformer sentiment scores.*

4. **Run the full pipeline (stance → sentiment → analysis) in one shot**:
   ```bash
   python scripts/run_pipeline.py \
     --parquet_dir data/parquet_sample \
     --out_dir ~/Documents/Reddit_EV_data_and_outputs/results \
     --log_level INFO
   ```
   *Writes stance and sentiment CSVs plus plots/tables in the `results/` folder.*

5. **Generate exploratory outputs manually (optional)**:
   ```bash
   python scripts/run_analysis.py \
     --stance_csv ~/Documents/Reddit_EV_data_and_outputs/results/stance_labels.csv \
     --sentiment_csv ~/Documents/Reddit_EV_data_and_outputs/results/sentiment_labels.csv \
     --out_dir ~/Documents/Reddit_EV_data_and_outputs/results/analysis \
     --timeframe monthly
   ```
   *Creates trend plots, sentiment boxplots, and a distinctive n-gram table under `results/analysis/` alongside an `analysis_manifest.json`.*

## Quick Test (≤3 rows)

Use these commands to sanity-check the pipeline quickly (replace `<dir>` with your parquet sample):

```bash
# stance + sentiment quick run
python scripts/quick_test.py --parquet_dir <dir>
    # defaults write to ~/Documents/Reddit_EV_data_and_outputs/quick_test

# tiny benchmark (prints timing + category breakdown)
python scripts/tiny_benchmark.py --parquet_dir <dir>

Outputs land in `results/quick_stance.csv` and `results/quick_sentiment.csv`. The benchmark reuses the same files.
```

Each command finishes in seconds, printing `elapsed=… rows/sec=…` summaries. The stance CSV contains subject/stance columns (`final_subject`, `final_stance`, `final_category`, `fused_pro`, `fused_anti`, `confidence`). The sentiment CSV contains only sentiment columns (`sent_vader_compound`, `sent_transformer_label`, `sent_transformer_score`).

## Optional Utilities
- `python scripts/run_vulture_report.py` – run Vulture dead-code analysis (saves `vulture_report.txt`).
- `scripts/to_csv.py` – convert a single `.zst` dump to CSV for inspection.
- `scripts/run_analysis.py` – rerun visualisations/analytics on existing CSVs.

## Repository Map
- `config/` – YAML files defining subreddits, keywords, negative filters, and labelling thresholds.
- `scripts/` – Command-line helpers (ingest, stance, sentiment, analysis, quick test, pipeline).
- `src/evrepo/` – Library modules for normalisation, filtering, subject scoring, weak rules, MNLI fusion, sentiment, etc.
- `tools/PushshiftDumps/` – Vendored utilities for reading `.zst` archives.

## Frequently Asked Questions
- **Why do some scripts take a long time the first run?** They download pretrained language models (hundreds of MB). Subsequent runs use cached copies.
- **Can I run without GPU?** Yes. The defaults use CPU-only inference; it's slower but works everywhere.
- **Where is my output?** Look inside the `results/` folder (stance/sentiment CSVs and plots) and `data/` for quick-test outputs.

## Citation
If you use this software in your research, please use the GitHub "Cite this repository" feature to generate the correct reference.

---

© 2025 OpenFis — Licensed under the MIT License (see `LICENSE`).
