# EV Reddit Stance Pipeline — Methods Overview

This document summarizes the end‑to‑end methodology for extracting EV‑related Reddit discussions, assigning stance categories, computing sentiment, and producing summary analyses.

## Data Source and Ingestion
- Data: Pushshift NDJSON `.zst` archives of Reddit submissions and comments (2022–2025).
- Scope: Selected liberal and conservative subreddits as specified in `config/subreddits.yaml` (no author‑level ideology inference).
- Ingestion: Streams every JSON object from `.zst` files and normalizes minimal fields: id, created_utc, subreddit, author, text, permalink, score, and an ideology group tag (from subreddit mapping).
- Partitioning: Records are written to Parquet partitions by `year=YYYY/month=MM/subreddit=NAME`.
- Time filtering: Applied per record using epoch timestamps derived from `created_utc` and user‑provided `--start`/`--end` bounds. Files may span years; filtering is record‑level.
- Language heuristic: Retains text that appears English based on character composition (rejects strings with very low alphabetic content or excessive non‑ASCII).
- Idempotency: Re‑runs append only unseen ids per partition; existing ids are skipped.

## EV Relevance Filtering
- Keywords: EV “product” core terms and context terms (e.g., electric vehicle, EV, BEV, PHEV; batteries/charging/range), mandate terms (e.g., mandate/ban/require), and non‑mandate policy terms (e.g., subsidies/tax credits/standards) defined in `config/keywords.yaml`.
- Negative filters: Regexes to exclude unrelated “EV” senses (e.g., expected value, electron‑volt, EVE Online) defined in `config/neg_filters.yaml`.
- Candidate criterion: A text passes when it contains product core phrases (and optionally context if configured), does not match any negative filters, and passes the English heuristic.

## Subject Scoring (Product / Mandate / Policy)
- For each text, raw hit counts are computed by regex against subject lists:
  - Product: core terms plus a down‑weighted contribution from context terms.
  - Mandate: mandate keyword hits.
  - Policy: non‑mandate policy hits (e.g., subsidies, standards).
- Normalization: A smooth saturating transform converts raw counts to [0,1] scores (1 − exp(−x)).
- Primary subject: Subject with maximum normalized score, with a deterministic tie‑break priority (mandate > product > policy; configurable).

## Stance Scoring per Subject (Simplified Default)
Default (fast, production): NLI‑only decision using a lightweight MNLI model.

1) Zero‑shot MNLI entailment (default)
- Model: `prajjwal1/bert-tiny-mnli` (fast). Subject‑specific hypotheses are evaluated for Pro and Anti.
- Decision rule: For the primary subject, compare (pro, anti); if max(pro, anti) < θ, assign Neutral (default θ = 0.55). Otherwise take the argmax.
- Output: Final category is subject × stance; confidence is |pro − anti|.

2) Optional enhancements (flag‑enabled)
- Large MNLI model: `facebook/bart-large-mnli` can be enabled via a CLI flag for improved accuracy at higher cost.
- Weak cues + fusion: Phrase‑based weak cues can be enabled; when on, stance uses a weighted fusion of weak cues and MNLI (w_weak=0.4, w_nli=0.6 by default).

## Sentiment Analysis
- VADER: Computes the standard compound score.
- Transformer: A modern classifier returns a discrete label and confidence (implementation uses a HuggingFace pipeline under the hood).
- Outputs: For each text id, the dataset stores VADER compound, transformer label, and transformer score.

## Aggregation and Visualization
- Time aggregation: Monthly or quarterly periods using UTC timestamps.
- Trend plots: Stacked areas or grouped bars per ideology showing proportions of the nine categories.
- Sentiment plots: Box/violin charts of VADER compound scores per ideology and final category.
- Distinctive n‑grams:
  - Within each (subject, ideology, stance) slice, collect 1–3‑grams (English stopwords removed).
  - Compute a log‑odds/Δlog‑probability style contrast (add‑one smoothing) versus the opposite stance; report top phrases for both Pro and Anti.

## Quality Control
- Sample review: Randomly sample ~200 high‑confidence cases (confidence ≥ 0.35) per ideology and stance.
- Adjustments: Expand weak cues or adjust θ (neutral threshold) if systematic mislabels are identified; re‑label affected partitions.

## Reproducibility and Reruns
- Idempotent writes: Ingestion appends only unseen ids per partition; labeling and sentiment skip ids already present in their outputs and deduplicate by id on write.
- Configuration dependence: Subreddit ideology map and keyword filters are externalized in YAML; analyses should report config versions and dates.
- Model selection: Report the NLI model used (`bart-large-mnli` vs `bert-tiny-mnli`) and any changes to fusion weights or neutral threshold.

## Outputs
- Parquet corpus (partitioned): `id, created_utc, subreddit, ideology_group, text, …`.
- Stance CSV: subject scores (raw/normalized), MNLI probabilities, weak cues (optional), final subject/stance/category, fused scores, confidence.
- Sentiment CSV: VADER compound, transformer label and score.
- Figures: trend plots (monthly/quarterly), sentiment distributions, and `distinctive_ngrams.csv`.

---

Notes
- While ingestion filters support time windows, `.zst` files are often whole‑history per subreddit; runtime is dominated by streaming and filtering large files.
- English detection is heuristic and intentionally conservative to avoid false positives from non‑English text.
