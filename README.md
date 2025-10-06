# EV Stance Reddit

Classify Reddit posts/comments into 9 categories (Pro/Against/Neutral × Product/Mandate/Policy) by ideology (liberal vs conservative), run sentiment, generate trends and distinctive n-grams, and export final CSV + per-category samples.

## Quick Start For Non-Specialists
1. **Install Python 3.10 or newer.** On Windows you can use the Microsoft Store build; on macOS/Linux use the official installer or `pyenv`.
2. **Clone the repository** and open a terminal in the project folder.
3. **Initialise dependencies** (choose one option):
   - Recommended: `poetry install`
   - Or plain `pip install -r` (see manual command below).
4. **Download Reddit Pushshift dumps** (the `.zst` files) into a folder you control, e.g. `C:\data\pushshift`.
5. **Run the sample pipeline** to ingest a small slice, label it, and inspect the CSV output (commands below). No prior data engineering experience is required—just copy the snippets into the terminal, one after another.

### Dependency Installation (Manual `pip` Alternative)
```
pip install polars pyarrow duckdb zstandard orjson pyyaml regex spacy vaderSentiment \
    transformers torch sentence-transformers typer[all] pydantic scikit-learn nltk \
    matplotlib plotly
```
*(The first run downloads ML models and may take several minutes.)*

## Step 1 – Ingest Example Data
This command scans Pushshift dumps, filters for EV conversations, and writes Parquet files locally.
```
python scripts/ingest_from_pushshiftdumps.py \
  --in_dir "C:\\data\\pushshift" \
  --out_parquet_dir data/parquet_sample \
  --start 2024-01-01 --end 2024-06-30 \
  --mode submissions \
  --subreddits AskALiberal \
  --max_records 2000 \
  --log_level INFO
```
You can change `--subreddits` or remove `--max_records` to process more content.

## Step 2 – Label & Score Sentiment (Produces a CSV Preview)
After ingestion, generate labels and sentiment scores and store them in `data/ev_labels_sample.csv` (allow a few minutes on CPU, especially the first time when models are downloaded):
```
python scripts/run_label_sample.py \
  --parquet_dir data/parquet_sample/year=2024 \
  --out_csv data/ev_labels_sample.csv \
  --limit 500 \
  --log_level INFO
```
What happens:
- **Subject scoring** (product vs mandate vs policy) uses keyword matches.
- **Weak rules** look for pro/anti phrases.
- **MNLI fusion** uses `facebook/bart-large-mnli` to infer stance (downloads on first use).
- **Sentiment analysis** adds VADER and DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`).

Open the resulting CSV in Excel/Sheets or run:
```
python -c "import pandas as pd; df=pd.read_csv('data/ev_labels_sample.csv'); print(df.head())"
```
to see the first rows, including columns such as `final_category`, `confidence`, `sent_vader_compound`, and `sent_transformer_label`.

## Optional – Convert Raw `.zst` Dumps to CSV
```
python scripts/to_csv.py \
  C:\\data\\pushshift\\AskALiberal_submissions.zst \
  data/csv_debug/AskALiberal_submissions.csv \
  --limit 500
```
Use this when you simply want to inspect the raw Pushshift files.

## Repository Map
- `config/` – YAML files defining subreddits, keywords, negative filters, and labelling thresholds.
- `scripts/` – Command-line helpers (`ingest_from_pushshiftdumps.py`, `run_label_sample.py`, `to_csv.py`).
- `src/evrepo/` – Library modules for normalisation, filtering, subject scoring, weak rules, MNLI fusion, sentiment, etc.
- `tools/PushshiftDumps/` – Vendored utilities for reading `.zst` archives.

## Frequently Asked Questions
- **Why do some scripts take a long time the first run?** They download pretrained language models (hundreds of MB). Subsequent runs use cached copies.
- **Can I run without GPU?** Yes. The defaults use CPU-only inference; it’s slower but works everywhere.
- **Where is my output?** Look inside the `data/` folder (Parquet partitions under `data/parquet*`, CSV previews directly under `data/`).

## Citation
If you use this software in your research, please use the GitHub “Cite this repository” feature to generate the correct reference.

---

© 2025 OpenFis — Licensed under the MIT License (see `LICENSE`).
