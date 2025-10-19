#!/usr/bin/env python
from __future__ import annotations

r"""
Analyze EV opinions: stance × ideology comparisons, sentiment stats, content analysis, and plots.

pip install pandas numpy scikit-learn matplotlib

python scripts/analyze_ev.py --stance_csv "C:\Users\awand\Documents\Reddit_EV_data_and_outputs\results\stance_labels.csv" --sentiment_csv "C:\Users\awand\Documents\Reddit_EV_data_and_outputs\results\sentiment_labels.csv" --out_dir "C:\Users\awand\Documents\Reddit_EV_data_and_outputs\results\analysis_out" --confidence_min 0.01 --min_docs_terms 30 --n_terms 20 --extra_stopwords "amp,im,don,t" --min_n_sentiment 10

"""


import argparse
import os
import json
import re
import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Robust column detection
# -------------------------
POSSIBLE_ID_COLUMNS = ["id", "item_id", "reddit_id", "post_id", "comment_id"]
POSSIBLE_TEXT_COLUMNS = ["body", "selftext", "title", "text", "content"]
POSSIBLE_CREATED_COLUMNS = ["created_utc", "created_ts", "created"]

def find_id_column(df: pd.DataFrame) -> str:
    for c in POSSIBLE_ID_COLUMNS:
        if c in df.columns:
            return c
    raise KeyError(f"No ID column found. Tried {POSSIBLE_ID_COLUMNS}. Available: {list(df.columns)}")

def find_text_column(df: pd.DataFrame) -> Optional[str]:
    for c in POSSIBLE_TEXT_COLUMNS:
        if c in df.columns:
            return c
    return None

def find_created_column(df: pd.DataFrame) -> Optional[str]:
    for c in POSSIBLE_CREATED_COLUMNS:
        if c in df.columns:
            return c
    return None

# -------------------------
# IO + merge (VADER sentiment; ideology filled from either file)
# -------------------------
def load_frames(stance_csv: str, sentiment_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load stance & sentiment CSVs, return (merged, stance_df, sent_df).

    Required columns per your examples:
      sentiment: id, sent_vader_compound, [ideology_group, subreddit, text, created_utc...]
      stance:    id, final_category, ideology_group (and confidence), [subreddit, text, created_utc...]
    """
    stance = pd.read_csv(stance_csv)
    sent = pd.read_csv(sentiment_csv)

    id_col_stance = find_id_column(stance)
    id_col_sent = find_id_column(sent)
    stance = stance.rename(columns={id_col_stance: "id"})
    sent = sent.rename(columns={id_col_sent: "id"})

    # Normalize ideology in BOTH frames (keep NaNs as NaN; .str preserves NaN)
    if "ideology_group" in stance.columns:
        stance["ideology_group"] = stance["ideology_group"].str.strip().str.lower()
    if "ideology_group" in sent.columns:
        sent["ideology_group"] = sent["ideology_group"].str.strip().str.lower()

    # Build numeric sentiment from VADER
    if "sent_vader_compound" not in sent.columns:
        raise KeyError("sentiment_labels.csv must include 'sent_vader_compound' for VADER-based analysis.")
    sent["sentiment_score"] = pd.to_numeric(sent["sent_vader_compound"], errors="coerce")

    # Merge AND bring ideology from both sides; then coalesce
    keep_cols = ["id", "sentiment_score", "sent_vader_compound"]
    if "ideology_group" in sent.columns:
        sent = sent.rename(columns={"ideology_group": "ideology_group_sent"})
        keep_cols.append("ideology_group_sent")

    merged = stance.merge(sent[[c for c in keep_cols if c in sent.columns]], on="id", how="left")

    if "ideology_group" not in merged.columns and "ideology_group_sent" in merged.columns:
        merged = merged.rename(columns={"ideology_group_sent": "ideology_group"})
    elif "ideology_group" in merged.columns and "ideology_group_sent" in merged.columns:
        # fill missing stance ideology from sentiment
        merged["ideology_group"] = merged["ideology_group"].fillna(merged["ideology_group_sent"])
        merged = merged.drop(columns=["ideology_group_sent"])

    # Required columns
    for col in ["final_category", "ideology_group"]:
        if col not in merged.columns:
            raise KeyError(f"Missing required column '{col}' in the merged data.")

    return merged, stance, sent

# -------------------------
# Sentiment summaries
# -------------------------
def _ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or (isinstance(std, float) and math.isnan(std)):
        return (np.nan, np.nan)
    se = std / math.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se

def summarize_sentiment_continuous(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Continuous mean/std/CI on sentiment_score (VADER in [-1,1])."""
    gb = df.groupby(["final_category", "ideology_group"])["sentiment_score"]
    stats = gb.agg(["count", "mean", "std"]).reset_index()
    stats = stats.rename(columns={"count": "n", "mean": "sent_mean", "std": "sent_std"})
    cis = stats.apply(lambda r: _ci95(r["sent_mean"], r["sent_std"], int(r["n"])), axis=1)
    stats["ci_low"] = [c[0] for c in cis]
    stats["ci_high"] = [c[1] for c in cis]
    stats["keep"] = stats["n"] >= min_n
    return stats.sort_values(["final_category", "ideology_group"]).reset_index(drop=True)

def add_binary_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Binary sentiment label as requested: >0 → positive, <0 → negative; ==0 stays NaN (ignored in binary rates)."""
    df = df.copy()
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    labels[df["sentiment_score"] > 0] = "positive"
    labels[df["sentiment_score"] < 0] = "negative"
    df["sentiment_label_bin"] = labels
    return df

# -------------------------
# Content analysis (TF-IDF)
# -------------------------
DEFAULT_STOPWORDS = set([
    "the","and","is","in","to","of","for","on","that","it","with","as","this","are","be","or","by","from","at",
    "an","was","but","have","has","if","not","you","they","he","she","we","i","a","so","do","does","did","will",
    "would","can","could","should","about","into","over","than","then","there","their","them","these","those",
    "ev","evs","electric","vehicle","vehicles","car","cars","tesla","battery","batteries","mandate","policy","policies"
])
def _ensure_text_column(df: pd.DataFrame) -> str:
    for c in POSSIBLE_TEXT_COLUMNS:
        if c in df.columns:
            return c
    raise KeyError(f"No text column found. Tried {POSSIBLE_TEXT_COLUMNS}. Available: {list(df.columns)}")

def _clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compute_top_terms_by_group(
    df: pd.DataFrame,
    min_docs: int = 30,
    n_terms: int = 20,
    ngram_range=(1, 2),
    extra_stopwords: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[tuple[str, float]]]]:
    """Return {final_category: {ideology_group: [(term, avg_weight), ...]}}."""
    text_col = _ensure_text_column(df)
    _df = df.copy()
    _df[text_col] = _df[text_col].astype(str).map(_clean_text)
    _df["pair"] = _df["final_category"].astype(str) + " | " + _df["ideology_group"].astype(str)

    texts = _df[text_col].tolist()
    groups = _df["pair"].tolist()

    stop = DEFAULT_STOPWORDS.copy()
    if extra_stopwords:
        stop.update([w.strip().lower() for w in extra_stopwords])

    vect = TfidfVectorizer(ngram_range=ngram_range, stop_words=list(stop), max_df=0.95, min_df=2)
    X = vect.fit_transform(texts)
    vocab = np.array(vect.get_feature_names_out())

    idx_by_group = pd.Series(range(len(groups))).groupby(pd.Series(groups)).apply(list)
    result_pair: Dict[str, List[tuple[str, float]]] = {}
    for g, idx in idx_by_group.items():
        if len(idx) < min_docs:
            continue
        sub = X[idx]
        weights = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = np.argsort(-weights)[:n_terms]
        result_pair[g] = list(zip(vocab[top_idx], weights[top_idx].tolist()))

    out: Dict[str, Dict[str, List[tuple[str, float]]]] = {}
    for pair, items in result_pair.items():
        cat, ideo = [x.strip() for x in pair.split("|")]
        out.setdefault(cat, {})[ideo] = items
    return out

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# -------------------------
# Plot helpers
# -------------------------
plt.rcParams.update({"figure.autolayout": True})

def _plot_counts_by_category_grouped_by_ideology(counts: pd.DataFrame, outdir: str, fname: str) -> str:
    if counts.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    pivot = counts.pivot_table(index="final_category", columns="ideology_group", values="n", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Counts by Final Category, grouped by Ideology")
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200); plt.close(fig); return out

def _plot_counts_by_ideology_grouped_by_category(counts: pd.DataFrame, outdir: str, fname: str) -> str:
    if counts.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    pivot = counts.pivot_table(index="ideology_group", columns="final_category", values="n", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Counts by Ideology, grouped by Final Category")
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200); plt.close(fig); return out

def _plot_props_within_ideology(counts: pd.DataFrame, outdir: str, fname: str) -> str:
    """Stacked proportions of categories within each ideology."""
    if counts.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    df = counts.groupby(["ideology_group", "final_category"])["n"].sum().reset_index()
    totals = df.groupby("ideology_group")["n"].transform("sum")
    df["share"] = df["n"] / totals
    pivot = df.pivot_table(index="ideology_group", columns="final_category", values="share", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6)); bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col].values, bottom=bottom, label=col); bottom += pivot[col].values
    ax.set_ylabel("Proportion within Ideology"); ax.set_title("Final Category proportions within each Ideology")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return out

def _plot_props_within_category(counts: pd.DataFrame, outdir: str, fname: str) -> str:
    """Stacked proportions of ideologies within each category."""
    if counts.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    df = counts.groupby(["final_category", "ideology_group"])["n"].sum().reset_index()
    totals = df.groupby("final_category")["n"].transform("sum")
    df["share"] = df["n"] / totals
    pivot = df.pivot_table(index="final_category", columns="ideology_group", values="share", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6)); bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col].values, bottom=bottom, label=col); bottom += pivot[col].values
    ax.set_ylabel("Proportion within Final Category"); ax.set_title("Ideology proportions within each Final Category")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); return out

def _plot_sentiment_heatmap(stats: pd.DataFrame, outdir: str, fname: str) -> str:
    if stats.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    pivot = stats.pivot_table(index="final_category", columns="ideology_group", values="sent_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_title("Mean VADER Sentiment (continuous) by Final Category × Ideology")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            label = "NA" if pd.isna(val) else f"{float(val):.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200); plt.close(fig); return out

def _plot_binary_rate(df: pd.DataFrame, outdir: str, fname: str) -> str:
    """Show positive rate by cell for binomial sentiment (>0)."""
    work = df.dropna(subset=["sentiment_label_bin"]).groupby(
        ["final_category","ideology_group"]
    )["sentiment_label_bin"].apply(lambda s: (s=="positive").mean()).reset_index(name="positive_rate")
    if work.empty: return ""
    os.makedirs(outdir, exist_ok=True)
    pivot = work.pivot_table(index="final_category", columns="ideology_group", values="positive_rate")
    fig, ax = plt.subplots(figsize=(10,6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Positive rate (>0)"); ax.set_title("Binary sentiment positive-rate by Final Category × Ideology")
    out = os.path.join(outdir, fname); fig.savefig(out, dpi=200); plt.close(fig); return out

# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze EV stances/sentiment/content and generate plots + report.")
    p.add_argument("--stance_csv", required=True, help="Output CSV from label_stance.py")
    p.add_argument("--sentiment_csv", required=True, help="Output CSV from score_sentiment.py")
    p.add_argument("--out_dir", required=True, help="Directory to write figures, tables, terms, report")
    p.add_argument("--confidence_col", default="confidence", help="Stance confidence column name")
    p.add_argument("--confidence_min", type=float, default=0.7, help="Minimum confidence to include in filtered plots")
    p.add_argument("--min_docs_terms", type=int, default=30, help="Minimum docs per (category, ideology) for top-terms")
    p.add_argument("--n_terms", type=int, default=20, help="Number of top terms to export per group")
    p.add_argument("--extra_stopwords", type=str, default="", help="Comma-separated extra stopwords for content analysis")
    p.add_argument("--min_n_sentiment", type=int, default=10, help="Min N per cell to include in sentiment summaries")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Output tree
    figs = os.path.join(args.out_dir, "figures")
    figs_filt = os.path.join(args.out_dir, f"figures_confidence_ge_{args.confidence_min:g}")
    tables = os.path.join(args.out_dir, "tables")
    terms_dir = os.path.join(args.out_dir, "terms")
    samples_dir = os.path.join(args.out_dir, "samples")
    reports_dir = os.path.join(args.out_dir, "reports")
    for d in [args.out_dir, figs, figs_filt, tables, terms_dir, samples_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    merged, stance_df, sent_df = load_frames(args.stance_csv, args.sentiment_csv)

    # Fill ideology from either file handled in load_frames; report if any still missing
    missing_ideo = merged["ideology_group"].isna().sum()
    if missing_ideo > 0:
        print(f"[WARN] {missing_ideo} rows still missing ideology_group — dropping for groupwise plots.")

    # Save merged to CSV
    merged_path = os.path.join(tables, "merged_labels_sentiment.csv")
    merged.to_csv(merged_path, index=False)

    # Prepare plotting dataframe
    plot_df = merged.dropna(subset=["ideology_group"]).copy()
    plot_df = add_binary_sentiment(plot_df)

    # 1) Counts per final_category × ideology_group (tables)
    counts = plot_df.groupby(["final_category", "ideology_group"]).size().reset_index(name="n")
    counts_path = os.path.join(tables, "counts_by_category_ideology.csv")
    counts.to_csv(counts_path, index=False)

    # 2) Per-subreddit counts → TXT
    txt_path = os.path.join(tables, "counts_by_category_ideology_per_subreddit.txt")
    if "subreddit" in plot_df.columns:
        sub_counts = (
            plot_df.groupby(["subreddit", "final_category", "ideology_group"])
            .size()
            .reset_index(name="n")
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            for (sr, cat, ideo), n in sub_counts.set_index(["subreddit", "final_category", "ideology_group"])["n"].items():
                f.write(f"{sr}\t{cat}\t{ideo}\t{n}\n")
    else:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# 'subreddit' column not available — no per-subreddit counts computed.\n")

    # 3) Sentiment summaries (continuous)
    sent_stats = summarize_sentiment_continuous(plot_df, min_n=args.min_n_sentiment)
    sent_stats_path = os.path.join(tables, "sentiment_summary_continuous.csv")
    sent_stats.to_csv(sent_stats_path, index=False)

    # 4) Plots — unfiltered
    _plot_counts_by_category_grouped_by_ideology(counts, figs, "counts_by_category__grouped_by_ideology.png")
    _plot_counts_by_ideology_grouped_by_category(counts, figs, "counts_by_ideology__grouped_by_category.png")
    _plot_props_within_ideology(counts, figs, "proportions_within_ideology.png")
    _plot_props_within_category(counts, figs, "proportions_within_category.png")
    _plot_sentiment_heatmap(sent_stats, figs, "sentiment_heatmap.png")
    _plot_binary_rate(plot_df, figs, "binary_positive_rate.png")

    # 5) Confidence-filtered variants (if column present)
    if args.confidence_col in plot_df.columns:
        filt = plot_df[plot_df[args.confidence_col] >= float(args.confidence_min)].copy()
        counts_f = filt.groupby(["final_category", "ideology_group"]).size().reset_index(name="n")
        sent_stats_f = summarize_sentiment_continuous(filt, min_n=args.min_n_sentiment)

        _plot_counts_by_category_grouped_by_ideology(counts_f, figs_filt, "counts_by_category__grouped_by_ideology.png")
        _plot_counts_by_ideology_grouped_by_category(counts_f, figs_filt, "counts_by_ideology__grouped_by_category.png")
        _plot_props_within_ideology(counts_f, figs_filt, "proportions_within_ideology.png")
        _plot_props_within_category(counts_f, figs_filt, "proportions_within_category.png")
        _plot_sentiment_heatmap(sent_stats_f, figs_filt, "sentiment_heatmap.png")
        _plot_binary_rate(filt, figs_filt, "binary_positive_rate.png")
    else:
        print(f"[INFO] Confidence column '{args.confidence_col}' not found; filtered plots skipped.")

    # 6) Content analysis (TF-IDF)
    extra_stop = [w.strip() for w in args.extra_stopwords.split(",") if w.strip()] if args.extra_stopwords else None
    try:
        terms = compute_top_terms_by_group(plot_df, min_docs=args.min_docs_terms, n_terms=args.n_terms,
                                           ngram_range=(1, 2), extra_stopwords=extra_stop)
        index: Dict[str, Dict[str, str]] = {}
        for cat, ideod in terms.items():
            for ideo, items in ideod.items():
                df_terms = pd.DataFrame(items, columns=["term", "avg_tfidf_weight"])
                fname = f"top_terms__{safe_name(cat)}__{safe_name(ideo)}.csv"
                path = os.path.join(terms_dir, fname)
                df_terms.to_csv(path, index=False)
                index.setdefault(cat, {})[ideo] = fname
        index_path = os.path.join(terms_dir, "top_terms_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        with open(os.path.join(terms_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

    # 7) Samples per final_category (30 rows each)
    if "final_category" in plot_df.columns:
        for cat, sub in plot_df.groupby("final_category"):
            sample = sub.sample(n=min(30, len(sub)), random_state=42)
            sample_path = os.path.join(samples_dir, f"sample__{safe_name(cat)}.csv")
            sample.to_csv(sample_path, index=False)

    # 8) Markdown report (short)
    report_md = os.path.join(reports_dir, "EV_opinions_report.md")
    lines: List[str] = []
    lines.append("# EV Opinions — Analysis Summary\n")
    lines.append("This report includes count & proportion comparisons by ideology/final_category, continuous VADER sentiment, binary rates, and confidence-filtered variants.\n")
    lines.append("## Data\n")
    lines.append(f"- Merged CSV: `{os.path.relpath(merged_path, reports_dir)}`")
    lines.append(f"- Counts: `{os.path.relpath(counts_path, reports_dir)}`")
    lines.append(f"- Sentiment summary (continuous): `{os.path.relpath(sent_stats_path, reports_dir)}`")
    lines.append("\n## Figures (unfiltered)\n")
    for fname in sorted(os.listdir(figs)):
        if fname.lower().endswith(".png"):
            lines.append(f"![{fname}]({os.path.join('..','figures',fname)})")
    if os.path.isdir(figs_filt):
        lines.append(f"\n## Figures (confidence ≥ {args.confidence_min:g})\n")
        for fname in sorted(os.listdir(figs_filt)):
            if fname.lower().endswith(".png"):
                lines.append(f"![{fname}]({os.path.join('..',os.path.basename(figs_filt),fname)})")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Analysis complete.")
    print(f"- Merged table: {merged_path}")
    print(f"- Counts table: {counts_path}")
    print(f"- Per-subreddit counts (txt): {txt_path}")
    print(f"- Sentiment summary (continuous): {sent_stats_path}")
    print(f"- Figures dir: {figs}")
    print(f"- Confidence-filtered figures dir: {figs_filt}")
    print(f"- Terms dir: {terms_dir}")
    print(f"- Samples dir: {samples_dir}")
    print(f"- Report: {report_md}")

if __name__ == "__main__":
    main()
