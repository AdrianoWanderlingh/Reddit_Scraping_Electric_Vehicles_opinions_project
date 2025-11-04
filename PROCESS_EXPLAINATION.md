# EV Reddit Stance Pipeline — Methods Overview and Variables Definition

This document summarizes the end‑to‑end methodology for extracting EV‑related Reddit discussions, assigning stance categories, computing sentiment, and producing summary analyses. The pipeline classifies Reddit posts and comments concerning electric vehicles (EVs) into **nine categories** (Pro/Against/Neutral for Product/Mandate/Policy subjects). These variables underpin the final report generation, trend plots, and diagnostic analysis.

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
- VADER: Computes the standard compound score. The Valence Aware Dictionary and sEntiment Reasoner is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. Note that sentiment polarity is itself not necessarily the most useful metric that exists.
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



# Data Dictionary and Variable Definitions

Here are the details relative to the variables generated by the EV Reddit Stance Analysis Pipeline, specifically those contained within the primary output files (`stance_labels.csv`, `sentiment_labels.csv`) and the merged data frame used for final aggregation and visualization.

---

## I. Core Metadata Variables (Ingested and Standardized)

These fields are present in both the Stance and Sentiment output files, originating from the initial Pushshift ingestion and filtering steps.

| Variable Name | Source File(s) | Definition | Contextual Information |
| :--- | :--- | :--- | :--- |
| **id** | Stance & Sentiment CSVs | The unique Reddit record identifier (submission or comment). | Essential for ensuring data idempotency and tracing records back to the original source. |
| **created\_utc** | Stance & Sentiment CSVs | The UTC epoch timestamp (integer) when the record was created. | Used for time filtering during ingestion and for aggregating data into **monthly or quarterly** periods for temporal trend plots. |
| **subreddit** | Stance & Sentiment CSVs | The name of the subreddit where the post originated. | Used to assign the `ideology_group` based on external YAML mapping (`config/subreddits.yaml`). |
| **ideology\_group** | Stance & Sentiment CSVs | The assigned political ideology tag: **'liberal' or 'conservative'**. | This is the main analytical grouping used throughout the pipeline to compare stance distributions and sentiment. |
| **is\_submission** | Stance & Sentiment CSVs | A boolean flag (True/False) indicating if the record is a submission (True) or a comment (False). | Used to differentiate record types; submissions combine `title` and `selftext` for analysis. |
| **record\_type** | Stance & Sentiment CSVs | User-friendly type: **'post'** (submission) or **'comment'**. | Generated during the labeling/scoring step based on `is_submission`. |
| **text** | Stance & Sentiment CSVs | The content used for scoring. | For submissions, this is composed by combining the normalized `title` and `selftext` fields. |
| **score** | Ingested Parquet (Merged) | The Reddit score (upvotes) of the record. | Unused as not extracted from the source currently, but could be used for the **Engagement Analysis** plot. |
| **year\_month** | Derived (Analysis) | The monthly time period derived from `created_utc`. | For plotting the **Temporal Evolution of Stances**. |
| **subject** | Derived (Analysis) | The standardized, lowercased primary subject: **'product', 'mandate', or 'policy'**. | Derived from `final_subject` during the merge step; used as the primary facet in analysis plots. |
| **stance** | Derived (Analysis) | The standardized, lowercased stance: **'pro', 'anti', or 'neutral'**. | Derived from `final_stance`; 'against' is converted to 'anti' for consistency. |

---

## II. Subject Scoring Variables (EV Focus Identification)

These variables capture the raw keyword counts and normalized scores for the three subjects defined in `config/keywords.yaml`.

| Variable Name | Definition | Contextual Information |
| :--- | :--- | :--- |
| **subject\_product\_raw** | Raw hit count of keywords specific to the Product subject. | Calculated as the sum of core term hits plus a **down-weighted contribution** from context terms ($\text{Core} + 0.5 \times \text{Context}$). |
| **subject\_product\_norm** | Normalized score for the Product subject. | Calculated using a smooth saturating transform: 1 - \exp(-x) (where x is the raw score), scaling the result to the range. |
| **subject\_mandate\_raw** | Raw hit count of keywords specific to the Mandate subject. | Based solely on hits against core mandate keyword lists (e.g., "mandate," "ban," "require"). |
| **subject\_mandate\_norm** | Normalized score for the Mandate subject. | Transformed using the same saturating function as the product score. |
| **subject\_policy\_raw** | Raw hit count of keywords specific to the Policy subject. | Based solely on hits against core policy keyword lists (e.g., "subsidies," "tax credits," "standards"). |
| **subject\_policy\_norm** | Normalized score for the Policy subject. | Transformed using the same saturating function as the product score. |
| **final\_subject** | The primary subject assigned to the text (product, mandate, or policy). | Determined by the subject with the **maximum normalized score**. Ties are broken using a deterministic priority (default: **mandate > product > policy**). |

---

## III. Stance Labeling Variables (NLI Classification)

These variables document the probabilities generated by the Natural Language Inference (NLI) model, the fusion mechanism, and the final decision.

| Variable Name | Definition | Contextual Information |
| :--- | :--- | :--- |
| **nli\_\*\_pro/anti** (e.g., `nli_product_pro`) | The NLI entailment probability (score) for the Pro or Anti hypothesis regarding a specific subject (Product, Mandate, or Policy). | These raw probabilities are generated by the zero-shot MNLI model (default: `prajjwal1/bert-tiny-mnli`). Scores are typically averaged over multiple **template paraphrases** (ensembling). |
| **fused\_pro/anti** | The final Pro/Anti score used to determine the stance. | If the optional **Weak Cues + Fusion** is disabled, these scores equal the raw NLI scores. If fusion is enabled, they are a weighted average of NLI and weak-cue scores (Wnli=0.6, Wweak=0.4 by default). |
| **final\_stance** | The assigned stance (Pro, Anti, or Neutral) for the `final_subject`. | Assigned based on `fused_pro` and `fused_anti`. If max(Pro,Anti) is below the **neutral threshold** the stance is **Neutral**. Default baseline theta = 0.55, currently used value as defined by `labeling.yaml` theta = 0.29. |
| **confidence** | The certainty of the final stance. | Calculated as the absolute difference between the final scores: `fused_pro` - `fused_anti` | **. High confidence (e.g., 0.35) is used for **Quality Control sampling**. |
| **final\_category** | The combined label resulting from the primary subject and the final stance (e.g., 'Pro-Mandate'). | This is one of the **nine final analysis categories**. |
| **weak\_\*\_pro/anti** (Optional) | Phrase-based weak cue scores for each subject. | These columns are only present if the `--use_weak_rules` flag is enabled. They are based on lexicon phrase hits (e.g., "range anxiety") and contribute to the `fused` scores. |

---

## IV. Sentiment Scoring Variables

These metrics quantify the emotional tone of the text, calculated independently of the ideological stance.

| Variable Name | Source File | Definition | Contextual Information |
| :--- | :--- | :--- | :--- |
| **sent\_vader\_compound** | `sentiment_labels.csv` | The **VADER compound score**. | VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based tool specifically tuned for social media sentiment. Scores range from -1 (negative) to +1 (positive). |
| **sent\_transformer\_label** | `sentiment_labels.csv` | The discrete sentiment label returned by the Transformer classifier. | Based on a pre-trained model (default: `distilbert-base-uncased-finetuned-sst-2-english`). |
| **sent\_transformer\_score** | `sentiment_labels.csv` | The confidence score associated with the Transformer label. | Used in visualization (e.g., Sentiment vs Stance hexbin plots) after being signed by its corresponding label. |

---

## V. Derived Variables (Used for Analysis and Diagnostics)

These variables are dynamically created during **Step 4 (Analysis and Plots)** (`analyze_ev.py.txt`) by merging the stance and sentiment data frames.

| Variable Name | Derivation | Contextual Information |
| :--- | :--- | :--- |
| **nli\_pro** | The NLI Pro score (`nli_*_pro`) specifically corresponding to the text’s **`final_subject`**. | Simplifies diagnostic plotting by holding the relevant Pro NLI score, regardless of whether the primary subject was Product, Mandate, or Policy. |
| **nli\_anti** | The NLI Anti score (`nli_*_anti`) specifically corresponding to the text’s **`final_subject`**. | Similarly, holds the relevant Anti NLI score for diagnostics, such as the NLI Score Scatter plot. |
| **signed\_confidence** | Derived by signing the `confidence` variable based on the `stance`: Pro is positive, Anti is negative, and Neutral is zero. | This variable is essential for the **Sentiment vs Stance Hexbin plots**, allowing visualization of whether positive sentiment correlates with Pro stance (positive Y-axis) or Anti stance (negative Y-axis). |

---

> This dictionary operates like a **database schema** for the final output. The `subject` and `stance` variables serve as the final, clean keys for organizing the data, while the various `raw`, `norm`, `nli`, and `fused` scores provide a comprehensive record of *how* the model arrived at the final classification, enabling reproducibility and diagnostic checks.