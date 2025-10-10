# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Generate trend plots, sentiment comparisons, and distinctive n-grams."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

SUBJECTS = ("product", "mandate", "policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots and tables from stance + sentiment CSVs")
    parser.add_argument("--stance_csv", required=True)
    parser.add_argument("--sentiment_csv", required=True)
    parser.add_argument("--out_dir", default="data/analysis")
    parser.add_argument("--timeframe", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap (useful for quick checks; max 3 recommended)")
    return parser.parse_args()


def load_data(stance_csv: Path, sentiment_csv: Path, limit: int | None) -> pd.DataFrame:
    stance_df = pd.read_csv(stance_csv)
    if limit:
        stance_df = stance_df.head(limit)
    stance_df["created_dt"] = pd.to_datetime(stance_df["created_utc"], unit="s", errors="coerce")
    stance_df["ideology_group"] = stance_df["ideology_group"].fillna("Unknown")
    sentiment_df = pd.read_csv(sentiment_csv)
    if limit:
        sentiment_df = sentiment_df.head(limit)
    sentiment_df = sentiment_df.add_suffix("_sent")
    sentiment_df = sentiment_df.rename(columns={"id_sent": "id"})
    merged = stance_df.merge(sentiment_df, on="id", how="left")
    return merged


def make_trend_plot(df: pd.DataFrame, out_dir: Path, timeframe: str) -> None:
    if timeframe == "quarterly":
        df["period"] = df["created_dt"].dt.to_period("Q").astype(str)
    else:
        df["period"] = df["created_dt"].dt.to_period("M").astype(str)
    agg = df.groupby(["period", "ideology_group", "final_category"]).size().reset_index(name="count")
    if agg.empty:
        return
    ideologies = sorted(agg["ideology_group"].unique())
    fig, axes = plt.subplots(len(ideologies), 1, figsize=(10, 3 * len(ideologies)), sharex=True)
    if len(ideologies) == 1:
        axes = [axes]
    for ax, ideology in zip(axes, ideologies):
        subset = agg[agg["ideology_group"] == ideology]
        pivot = subset.pivot(index="period", columns="final_category", values="count").fillna(0)
        pivot.plot(kind="area", stacked=True, ax=ax)
        ax.set_title(f"{ideology} - category distribution ({timeframe})")
        ax.set_ylabel("count")
    axes[-1].set_xlabel("period")
    out_path = out_dir / f"trend_{timeframe}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_sentiment_plot(df: pd.DataFrame, out_dir: Path) -> None:
    if "sent_vader_compound_sent" not in df:
        return
    subset = df.dropna(subset=["sent_vader_compound_sent"])
    if subset.empty:
        return
    ideologies = sorted(subset["ideology_group"].unique())
    fig, axes = plt.subplots(len(ideologies), 1, figsize=(10, 3 * len(ideologies)), sharex=True)
    if len(ideologies) == 1:
        axes = [axes]
    for ax, ideology in zip(axes, ideologies):
        part = subset[subset["ideology_group"] == ideology]
        categories = sorted(part["final_category"].unique())
        data = [part[part["final_category"] == cat]["sent_vader_compound_sent"].dropna() for cat in categories]
        if any(len(series) for series in data):
            ax.boxplot(data, labels=categories, showmeans=True)
        ax.set_title(f"Sentiment distribution - {ideology}")
        ax.set_ylabel("VADER compound")
        ax.tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("final category")
    out_path = out_dir / "sentiment_boxplot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_distinctive(df: pd.DataFrame, out_dir: Path) -> None:
    rows: List[Dict[str, str | float]] = []
    for subject in SUBJECTS:
        subject_df = df[df["final_subject"] == subject]
        if subject_df.empty:
            continue
        ideologies = subject_df["ideology_group"].unique()
        for ideology in ideologies:
            part = subject_df[subject_df["ideology_group"] == ideology]
            pro_texts = part[part["final_stance"].str.lower() == "pro"]["text"].dropna().astype(str)
            anti_texts = part[part["final_stance"].str.lower() == "anti"]["text"].dropna().astype(str)
            if len(pro_texts) == 0 or len(anti_texts) == 0:
                continue
            vectorizer = CountVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                min_df=1,
                stop_words="english",
            )
            corpus = pro_texts.tolist() + anti_texts.tolist()
            X = vectorizer.fit_transform(corpus)
            vocab = vectorizer.get_feature_names_out()
            pro_count = np.asarray(X[: len(pro_texts)].sum(axis=0)).flatten() + 1
            anti_count = np.asarray(X[len(pro_texts) :].sum(axis=0)).flatten() + 1
            pro_prob = pro_count / pro_count.sum()
            anti_prob = anti_count / anti_count.sum()
            log_ratio = np.log(pro_prob) - np.log(anti_prob)
            pro_indices = np.argsort(-log_ratio)[:5]
            anti_indices = np.argsort(log_ratio)[:5]
            for idx in pro_indices:
                rows.append(
                    {
                        "subject": subject,
                        "ideology_group": ideology,
                        "stance": "pro",
                        "ngram": vocab[idx],
                        "score": float(log_ratio[idx]),
                    }
                )
            for idx in anti_indices:
                rows.append(
                    {
                        "subject": subject,
                        "ideology_group": ideology,
                        "stance": "anti",
                        "ngram": vocab[idx],
                        "score": float(-log_ratio[idx]),
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "distinctive_ngrams.csv", index=False)


def run_analysis(stance_csv: str | Path, sentiment_csv: str | Path, out_dir: str | Path, timeframe: str = "monthly", limit: int | None = None) -> None:
    stance_path = Path(stance_csv)
    sentiment_path = Path(sentiment_csv)
    analysis_dir = Path(out_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(stance_path, sentiment_path, limit)
    make_trend_plot(df, analysis_dir, timeframe)
    make_sentiment_plot(df, analysis_dir)
    compute_distinctive(df, analysis_dir)

    manifest = {
        "stance_csv": str(stance_path),
        "sentiment_csv": str(sentiment_path),
        "timeframe": timeframe,
        "limit": limit,
        "outputs": [
            f"trend_{timeframe}.png",
            "sentiment_boxplot.png",
            "distinctive_ngrams.csv",
        ],
    }
    (analysis_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.stance_csv, args.sentiment_csv, args.out_dir, timeframe=args.timeframe, limit=args.limit)
