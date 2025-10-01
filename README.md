# EV Stance Reddit

Classify Reddit posts/comments into 9 categories (Pro/Against/Neutral × Product/Mandate/Policy) by ideology (liberal vs conservative), run sentiment, generate trends and distinctive n-grams, and export final CSV + per-category samples.

## Getting Started
- Install a recent Python (>=3.10). Poetry is recommended for dependency management, or install requirements manually using `pip`.
- Ensure the PushshiftDumps git submodule is initialised: `git submodule update --init --recursive`.
- Download the Pushshift `.zst` archives you plan to ingest and place them somewhere accessible (e.g. `C:\data\pushshift`).

### Install Dependencies with Poetry
```
poetry install
```

Or install the core runtime packages manually:
```
pip install polars pyarrow duckdb zstandard orjson pyyaml regex spacy vaderSentiment \
    transformers torch sentence-transformers typer[all] pydantic scikit-learn nltk \
    matplotlib plotly
```

## Ingesting Pushshift Dumps
Use `scripts/ingest_from_pushshiftdumps.py` to stream `.zst` archives, filter EV-related rows, and write partitioned Parquet files.

### Typical Full Run
```
python scripts/ingest_from_pushshiftdumps.py \
  --in_dir "C:\\data\\pushshift" \
  --out_parquet_dir data/parquet \
  --start 2018-01-01 --end 2025-06-30 \
  --mode both \
  --ideology_map config/subreddits.yaml \
  --keywords config/keywords.yaml \
  --neg_filters config/neg_filters.yaml \
  --log_level INFO
```

### Sample Run: Single Subreddit, Six Months, Limited Rows
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
This command narrows processing to `AskALiberal` submissions in the first half of 2024 and stops after 2,000 matching EV rows—ideal for dry runs or quick QA.

### Optional: Raw CSV Debug Dumps
Provide `--out_csv_dir` to mirror each `.zst` as a CSV using the Pushshift helper scripts. For example:
```
python scripts/ingest_from_pushshiftdumps.py \
  --in_dir "C:\\data\\pushshift" \
  --out_parquet_dir data/parquet \
  --out_csv_dir data/csv_debug \
  --mode comments
```

## Converting a Single `.zst` to CSV
`scripts/to_csv.py` converts one archive to a CSV for quick review or sharing. This does not filter for keywords or else.
```
  python scripts/to_csv.py \
  C:/Users/awand/Desktop/reddit_pushshift_dump/subreddits24/Anarcho_Capitalism_comments.zst \
  data/test_fixture/test_sample_preview.csv \
  --limit 100
```
- Use `--fields` to override the default header.
- `--limit` lets you sample a manageable number of rows from very large files.

## Repository Structure
- `config/`: YAML files that define subreddits, EV keywords, negative filters, and future labelling thresholds.
- `scripts/`: Utility entry points for ingesting data (`ingest_from_pushshiftdumps.py`) and ad-hoc CSV export (`to_csv.py`).
- `src/evrepo/`: Library modules shared across the pipeline (normalisation, filtering, utilities).
- `tools/PushshiftDumps/`: Vendored helper scripts from Watchful1 for reading `.zst` archives.

## Next Steps
After ingesting data into `data/parquet`, subsequent stages (labelling, sentiment, aggregation, and visualisation) will consume those partitions. Follow the roadmap in `AGENTS.md` as new modules are implemented.

## Citation
If you use this software in your research, please use the **"Cite this repository"** feature on GitHub (right-hand side of this page) to generate the correct reference.

---

© 2025 OpenFis — Licensed under the MIT License. See `LICENSE` for details.
