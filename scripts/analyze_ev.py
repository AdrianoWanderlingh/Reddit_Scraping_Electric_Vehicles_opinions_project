#!/usr/bin/env python
from __future__ import annotations
from turtle import color
from pathlib import Path


r"""
Comprehensive EV opinions analysis: Essential Core Plots + Advanced Analytics + Diagnostics.

Includes:
- Stance distribution by ideology (stacked bars per subject)
- Temporal evolution (time series by ideology × stance)
- Subject classification flow (Sankey diagram)
- Confidence distributions (violin plots)
- Sentiment vs Stance scatter (VADER + Transformer)
- Subreddit-level heatmap
- Engagement/score analysis
- Classification diagnostics

pip install pandas numpy scikit-learn matplotlib seaborn plotly

python scripts/analyze_ev_enhanced.py \
    --stance_csv "path/to/stance_labels.csv" \
    --sentiment_csv "path/to/sentiment_labels.csv" \
    --out_dir "path/to/analysis_out" \
    --confidence_min 0.01
"""

import argparse
import os
import re
import math
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Try plotly for Sankey; graceful fallback if missing
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARN] plotly not available; Sankey diagram will be skipped.")

# -------------------------
# Style setup
# -------------------------
plt.rcParams.update({
    "figure.autolayout": True,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 10,
})
sns.set_palette("Set2")

# --- YAML mapping support for subreddit -> ideology
try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

def _norm_subreddit(name: str) -> str:
    """Normalize names from CSV ('Futurology') and YAML ('r/Futurology') to 'futurology'."""
    s = str(name or "").strip().lower()
    if s.startswith("r/"):
        s = s[2:]
    return s

def _load_sub_map_from_yaml() -> Dict[str, str]:
    """
    Reads config/subreddits.yaml (or .yml / .json) with shape:
      liberal: [r/politics, ...]
      conservative: [r/Conservative, ...]
    Returns { 'politics': 'liberal', 'conservative': 'conservative', ... }
    """
    candidates = ("config/subreddits.yaml", "config/subreddits.yml", "config/subreddits.json", "subreddits.yaml")
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        print("[WARN] No subreddits.yaml found; ideology_group may be NaN.")
        return {}

    try:
        if path.endswith(".json"):
            import json
            raw = json.load(open(path, "r", encoding="utf-8"))
        else:
            if not YAML_AVAILABLE:
                print("[WARN] PyYAML not installed; cannot read YAML. `pip install pyyaml`")
                return {}
            raw = yaml.safe_load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {}

    sub_map: Dict[str, str] = {}
    for ideo in ("liberal", "conservative"):
        for sub in (raw.get(ideo) or []):
            sub_map[_norm_subreddit(sub)] = ideo  # add 'politics' from 'r/politics'
    # also add keys that still include 'r/' just in case
    sub_map.update({f"r/{k}": v for k, v in list(sub_map.items())})
    return sub_map


# -------------------------
# Robust column detection
# -------------------------
POSSIBLE_ID_COLUMNS = ["id", "item_id", "reddit_id", "post_id", "comment_id"]
POSSIBLE_TEXT_COLUMNS = ["body", "selftext", "title", "text", "content"]
POSSIBLE_CREATED_COLUMNS = ["created_utc", "created_ts", "created"]
POSSIBLE_SCORE_COLUMNS = ["score", "ups", "upvotes"]
POSSIBLE_SUBREDDIT_COLUMNS = ["subreddit", "sub", "community"]

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def require_id_column(df: pd.DataFrame) -> str:
    col = find_column(df, POSSIBLE_ID_COLUMNS)
    if col is None:
        raise KeyError(f"No ID column found. Tried {POSSIBLE_ID_COLUMNS}. Available: {list(df.columns)}")
    return col

# -------------------------
# IO + merge
# -------------------------
def load_frames(stance_csv: str, sentiment_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and merge stance + sentiment CSVs with robust parsing (no hard dependency on 'score')."""
    stance = pd.read_csv(stance_csv)
    sent = pd.read_csv(sentiment_csv)

    # Normalize IDs
    stance = stance.rename(columns={require_id_column(stance): "id"})
    sent   = sent.rename(columns={require_id_column(sent): "id"})

    # Normalize ideology in both (turn empty strings into NaN)
    for df in (stance, sent):
        if "ideology_group" in df.columns:
            df["ideology_group"] = (
                df["ideology_group"].astype(str).str.strip().str.lower().replace({"": np.nan})
            )

    # Sentiment fields
    if "sent_vader_compound" not in sent.columns:
        raise KeyError("sentiment_labels.csv must include 'sent_vader_compound'")
    sent["sentiment_vader"] = pd.to_numeric(sent["sent_vader_compound"], errors="coerce")
    if "sent_transformer_label" in sent.columns:
        sent["sentiment_transformer_label"] = sent["sent_transformer_label"].astype(str).str.lower()
    if "sent_transformer_score" in sent.columns:
        sent["sentiment_transformer_score"] = pd.to_numeric(sent["sent_transformer_score"], errors="coerce")

    # Merge
    keep_cols = ["id", "sentiment_vader"]
    if "ideology_group" in sent.columns:
        sent = sent.rename(columns={"ideology_group": "ideology_group_sent"})
        keep_cols.append("ideology_group_sent")
    if "sentiment_transformer_label" in sent.columns:
        keep_cols.append("sentiment_transformer_label")
    if "sentiment_transformer_score" in sent.columns:
        keep_cols.append("sentiment_transformer_score")

    merged = stance.merge(sent[[c for c in keep_cols if c in sent.columns]], on="id", how="left")

    # Coalesce ideology
    if "ideology_group" not in merged.columns and "ideology_group_sent" in merged.columns:
        merged = merged.rename(columns={"ideology_group_sent": "ideology_group"})
    elif "ideology_group" in merged.columns and "ideology_group_sent" in merged.columns:
        merged["ideology_group"] = merged["ideology_group"].fillna(merged["ideology_group_sent"])
        merged = merged.drop(columns=["ideology_group_sent"])

    # Subject & stance from explicit columns if present, else parse from final_category
    if "final_subject" in merged.columns:
        merged["subject"] = merged["final_subject"].astype(str).str.strip().str.lower()
    elif "final_category" in merged.columns:
        merged["subject"] = merged["final_category"].str.extract(r"\((product|mandate|policy)\)", expand=False).str.lower()
    else:
        merged["subject"] = np.nan

    if "final_stance" in merged.columns:
        merged["stance"] = merged["final_stance"].astype(str).str.strip().str.lower()
    elif "final_category" in merged.columns:
        merged["stance"] = (
            merged["final_category"]
            .str.extract(r"^(Pro|Against|Neutral)", expand=False)
            .str.lower()
            .replace({"against": "anti"})
        )
    else:
        merged["stance"] = np.nan

    # Created_utc → datetime
    created_col = find_column(merged, POSSIBLE_CREATED_COLUMNS)
    if created_col:
        merged["created_utc"] = pd.to_numeric(merged[created_col], errors="coerce")
        merged["datetime"] = pd.to_datetime(merged["created_utc"], unit="s", errors="coerce")
        merged["year_month"] = merged["datetime"].dt.to_period("M")

    # Score/ups might not exist in your data; only map if found
    score_col = find_column(merged, POSSIBLE_SCORE_COLUMNS)
    if score_col and score_col != "score":
        merged = merged.rename(columns={score_col: "score"})
    if "score" in merged.columns:
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")

        # Subreddit name normalization
    sub_col = find_column(merged, POSSIBLE_SUBREDDIT_COLUMNS)
    if sub_col and sub_col != "subreddit":
        merged = merged.rename(columns={sub_col: "subreddit"})

    # --- Ideology mapping from YAML (authoritative)
    sub_map = _load_sub_map_from_yaml()
    if "subreddit" in merged.columns and sub_map:
        merged["__sub_norm"] = merged["subreddit"].apply(_norm_subreddit)
        merged["ideology_group"] = merged["__sub_norm"].map(sub_map)

        # Report any misses clearly
        misses = sorted(merged.loc[merged["ideology_group"].isna(), "__sub_norm"].dropna().unique())
        if misses:
            print(f"[WARN] Unmapped subreddits in subreddits.yaml: {misses}")
            # If you want to fail hard instead, uncomment:
            # raise ValueError(f"Unmapped subreddits: {misses}")

        # Optional: enforce only 'liberal'/'conservative'
        merged["ideology_group"] = merged["ideology_group"].where(
            merged["ideology_group"].isin(["liberal", "conservative"]), np.nan
        )

        merged = merged.drop(columns=["__sub_norm"], errors="ignore")

    # Confidence numeric
    if "confidence" in merged.columns:
        merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")

    # Subject-specific NLI → generic columns for diagnostics
    subject_to_cols = {
        "product": ("nli_product_pro", "nli_product_anti"),
        "mandate": ("nli_mandate_pro", "nli_mandate_anti"),
        "policy":  ("nli_policy_pro",  "nli_policy_anti"),
    }
    merged["nli_pro"] = np.nan
    merged["nli_anti"] = np.nan
    if any(col in merged.columns for cols in subject_to_cols.values() for col in cols):
        for subj, (pro_col, anti_col) in subject_to_cols.items():
            mask = merged["subject"].eq(subj)
            if pro_col in merged.columns:
                merged.loc[mask, "nli_pro"]  = pd.to_numeric(merged.loc[mask, pro_col],  errors="coerce")
            if anti_col in merged.columns:
                merged.loc[mask, "nli_anti"] = pd.to_numeric(merged.loc[mask, anti_col], errors="coerce")

    # Validate values
    merged.loc[~merged["subject"].isin(["product", "mandate", "policy"]), "subject"] = np.nan
    merged.loc[~merged["stance"].isin(["pro", "anti", "neutral"]), "stance"] = np.nan

    return merged, stance, sent


# -------------------------
# Utility functions
# -------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def _ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 1 or pd.isna(std):
        return (np.nan, np.nan)
    se = std / math.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se

STANCE_COLORS = {"pro": "#2ecc71", "anti": "#e74c3c", "neutral": "#95a5a6"}

# -------------------------
# ESSENTIAL CORE PLOTS
# -------------------------

def plot_stance_distribution_by_ideology(df: pd.DataFrame, outdir: str) -> str:
    """1. Stance Distribution by Ideology Group - stacked bars by subject."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["ideology_group", "stance", "subject"]).copy()
    if work.empty:
        return ""

    subjects = sorted(work["subject"].unique())
    fig, axes = plt.subplots(1, len(subjects), figsize=(5*len(subjects), 6), squeeze=False)
    axes = axes.flatten()

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subdf = work[work["subject"] == subj]

        counts = subdf.groupby(["ideology_group", "stance"]).size().reset_index(name="n")
        totals = counts.groupby("ideology_group")["n"].transform("sum")
        counts["prop"] = counts["n"] / totals

        # Ensure all stances exist
        for st in ["pro", "anti", "neutral"]:
            if st not in counts["stance"].unique():
                counts = pd.concat([counts, pd.DataFrame([{"ideology_group": ig, "stance": st, "n": 0, "prop": 0.0}
                                                          for ig in counts["ideology_group"].unique()])], ignore_index=True)

        pivot = counts.pivot_table(index="ideology_group", columns="stance", values="prop", fill_value=0)

        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=[STANCE_COLORS.get(c, "#777777") for c in pivot.columns])
        ax.set_title(f"{subj.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ideology Group", fontsize=10)
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Stance Distribution by Ideology (faceted by Subject)",
                 fontsize=14, fontweight="bold", y=1.02)
    out = os.path.join(outdir, "1_stance_distribution_by_ideology.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_stance_distribution_by_ideology_no_neutral(df: pd.DataFrame, outdir: str) -> str:
    """1b. Stance Distribution by Ideology (no Neutral) - stacked bars per subject."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["ideology_group", "stance", "subject"]).copy()
    work = work[work["stance"].isin(["pro", "anti"])]
    if work.empty:
        return ""

    subjects = sorted(work["subject"].unique())
    fig, axes = plt.subplots(1, len(subjects), figsize=(5*len(subjects), 6), squeeze=False)
    axes = axes.flatten()

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subdf = work[work["subject"] == subj]

        counts = subdf.groupby(["ideology_group", "stance"]).size().reset_index(name="n")
        totals = counts.groupby("ideology_group")["n"].transform("sum")
        counts["prop"] = counts["n"] / totals

        # Ensure both stances exist
        for st in ["pro", "anti"]:
            if st not in counts["stance"].unique():
                counts = pd.concat([counts, pd.DataFrame([{"ideology_group": ig, "stance": st, "n": 0, "prop": 0.0}
                                                          for ig in counts["ideology_group"].unique()])],
                                   ignore_index=True)

        pivot = counts.pivot_table(index="ideology_group", columns="stance", values="prop", fill_value=0)
        pivot = pivot[["pro", "anti"]] if set(["pro", "anti"]).issubset(pivot.columns) else pivot

        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=[STANCE_COLORS["pro"], STANCE_COLORS["anti"]])
        ax.set_title(f"{subj.capitalize()} (No Neutral)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ideology Group", fontsize=10)
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Stance Distribution by Ideology (No Neutral; faceted by Subject)",
                 fontsize=14, fontweight="bold", y=1.02)
    out = os.path.join(outdir, "1b_stance_distribution_by_ideology_no_neutral.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out



def plot_temporal_evolution(df: pd.DataFrame, outdir: str) -> str:
    """2. Temporal Evolution - multi-line time series by ideology × stance, with clear legends."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["year_month", "ideology_group", "stance", "subject"]).copy()
    if work.empty or "year_month" not in work.columns:
        return ""

    subjects = sorted(work["subject"].unique())
    fig, axes = plt.subplots(len(subjects), 1, figsize=(14, 5*len(subjects)), squeeze=False)
    axes = axes.flatten()

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subdf = work[work["subject"] == subj]

        monthly = subdf.groupby(["year_month", "ideology_group", "stance"]).size().reset_index(name="n")
        totals = monthly.groupby(["year_month", "ideology_group"])["n"].transform("sum")
        monthly["prop"] = monthly["n"] / totals
        monthly["date"] = monthly["year_month"].dt.to_timestamp()

        # Plot lines: no labels per-line; we'll add clean, manual legends
        for ideo in sorted(monthly["ideology_group"].unique()):
            for stance in ["pro", "anti", "neutral"]:
                subset = monthly[(monthly["ideology_group"] == ideo) & (monthly["stance"] == stance)]
                if subset.empty:
                    continue
                linestyle = "-" if ideo == "liberal" else "--"
                color = STANCE_COLORS.get(stance, "#777777")
                marker = "o" if ideo == "liberal" else "s"
                ax.plot(subset["date"], subset["prop"],
                        linestyle=linestyle, color=color, linewidth=2,
                        marker=marker, markersize=3, alpha=0.9)

        ax.set_title(f"{subj.capitalize()} Stances Over Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Proportion within Ideology", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Build two separate legends: stance colors and ideology styles
        stance_handles = [
            Line2D([0], [0], color=STANCE_COLORS["pro"], lw=2, label="Pro"),
            Line2D([0], [0], color=STANCE_COLORS["anti"], lw=2, label="Anti"),
            Line2D([0], [0], color=STANCE_COLORS["neutral"], lw=2, label="Neutral"),
        ]
        ideology_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", label="Liberal (solid)"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", label="Conservative (dashed)"),
        ]
        leg1 = ax.legend(handles=stance_handles, title="Stance", loc="upper left")
        ax.add_artist(leg1)
        ax.legend(handles=ideology_handles, title="Ideology style", loc="upper right")

    plt.suptitle("Temporal Evolution of Stances by Ideology", fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "2_temporal_evolution.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_sankey_flow(df: pd.DataFrame, outdir: str) -> str:
    """3. Subject Classification Flow - Sankey diagram."""
    if not PLOTLY_AVAILABLE:
        return ""

    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["ideology_group", "subject", "stance"]).copy()
    if work.empty:
        return ""

    flow = work.groupby(["ideology_group", "subject", "stance"]).size().reset_index(name="value")

    ideologies = sorted(flow["ideology_group"].unique())
    subjects = sorted(flow["subject"].unique())
    stances = ["pro", "anti", "neutral"]  # fixed order for consistent coloring

    nodes = ideologies + subjects + stances
    node_dict = {n: i for i, n in enumerate(nodes)}

    sources, targets, values = [], [], []
    # Ideology -> Subject
    for _, row in flow.groupby(["ideology_group", "subject"])["value"].sum().reset_index().iterrows():
        sources.append(node_dict[row["ideology_group"]]); targets.append(node_dict[row["subject"]]); values.append(int(row["value"]))
    # Subject -> Stance
    for _, row in flow.iterrows():
        sources.append(node_dict[row["subject"]]); targets.append(node_dict[row["stance"]]); values.append(int(row["value"]))

    node_colors = (
        ["#6baed6"] * len(ideologies) +
        ["#bcbddc"] * len(subjects) +
        [STANCE_COLORS["pro"], STANCE_COLORS["anti"], STANCE_COLORS["neutral"]]
    )

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=values)
    )])

    fig.update_layout(title_text="EV Discussion Flow: Ideology → Subject → Stance", font_size=12, height=600)
    out = os.path.join(outdir, "3_sankey_flow.html")
    fig.write_html(out)
    return out

def plot_confidence_distributions(df: pd.DataFrame, outdir: str) -> str:
    """4. Confidence Distribution - violin plots."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["confidence", "ideology_group", "stance", "subject"]).copy()
    if work.empty:
        return ""

    subjects = sorted(work["subject"].unique())
    fig, axes = plt.subplots(1, len(subjects), figsize=(6*len(subjects), 6), squeeze=False)
    axes = axes.flatten()

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subdf = work[work["subject"] == subj]
        sns.violinplot(data=subdf, x="ideology_group", y="confidence",
                       hue="stance", split=False, ax=ax, inner="quartile",
                       palette=STANCE_COLORS)
        ax.set_title(f"{subj.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ideology Group", fontsize=10)
        ax.set_ylabel("Stance Confidence", fontsize=10)
        ax.legend(title="Stance", loc="lower right")

    plt.suptitle("Stance Confidence Distributions",
                 fontsize=14, fontweight="bold", y=1.02)
    out = os.path.join(outdir, "4_confidence_distributions.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

# -------------------------
# ADVANCED ANALYTICAL PLOTS
# -------------------------

def plot_sentiment_stance_vader(df: pd.DataFrame, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["sentiment_vader", "confidence", "stance", "ideology_group"]).copy()
    if work.empty:
        return ""

    work["signed_confidence"] = work["confidence"]
    work.loc[work["stance"] == "anti", "signed_confidence"] *= -1
    work.loc[work["stance"] == "neutral", "signed_confidence"] = 0

    # Dynamic y range
    y_min = float(work["signed_confidence"].min())
    y_max = float(work["signed_confidence"].max())
    pad = max(0.01, 0.05 * (y_max - y_min if y_max > y_min else 0.1))
    y_lo, y_hi = y_min - pad, y_max + pad

    ideos = sorted(work["ideology_group"].dropna().unique())
    fig, axes = plt.subplots(1, len(ideos), figsize=(7*len(ideos), 6))
    if len(ideos) == 1:
        axes = [axes]

    for ax, ideo in zip(axes, ideos):
        subdf = work[work["ideology_group"] == ideo]
        hb = ax.hexbin(subdf["sentiment_vader"], subdf["signed_confidence"],
                       gridsize=30, cmap="YlOrRd", mincnt=1, alpha=0.8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_title(f"{ideo.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("VADER Sentiment Score", fontsize=10)
        ax.set_ylabel("Signed Stance Confidence\n(+Pro, 0 Neutral, -Anti)", fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(y_lo, y_hi)
        plt.colorbar(hb, ax=ax, label="Count")

    plt.suptitle("Sentiment vs Stance Confidence (VADER)", fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "5a_sentiment_stance_vader.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
    """5a. Sentiment vs Stance (VADER) - scatter/hexbin."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["sentiment_vader", "confidence", "stance", "ideology_group"]).copy()
    if work.empty:
        return ""

    work["signed_confidence"] = work["confidence"]
    work.loc[work["stance"] == "anti", "signed_confidence"] *= -1
    work.loc[work["stance"] == "neutral", "signed_confidence"] = 0

    ideos = sorted(work["ideology_group"].dropna().unique())
    fig, axes = plt.subplots(1, len(ideos), figsize=(7*len(ideos), 6))

    if len(ideos) == 1:
        axes = [axes]

    for ax, ideo in zip(axes, ideos):
        subdf = work[work["ideology_group"] == ideo]
        hb = ax.hexbin(subdf["sentiment_vader"], subdf["signed_confidence"],
                       gridsize=30, cmap="YlOrRd", mincnt=1, alpha=0.8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_title(f"{ideo.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("VADER Sentiment Score", fontsize=10)
        ax.set_ylabel("Signed Stance Confidence\n(+Pro, 0 Neutral, -Anti)", fontsize=10)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        plt.colorbar(hb, ax=ax, label="Count")

    plt.suptitle("Sentiment vs Stance Confidence (VADER)",
                 fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "5a_sentiment_stance_vader.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_sentiment_stance_transformer(df: pd.DataFrame, outdir: str) -> str:
    """5b. Sentiment vs Stance (Transformer) - scatter/hexbin."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["sentiment_transformer_score", "sentiment_transformer_label",
                             "confidence", "stance", "ideology_group"]).copy()
    if work.empty:
        return ""

    work["transformer_signed"] = work["sentiment_transformer_score"]
    work.loc[work["sentiment_transformer_label"].str.contains("neg", case=False, na=False),
             "transformer_signed"] *= -1

    work["signed_confidence"] = work["confidence"]
    work.loc[work["stance"] == "anti", "signed_confidence"] *= -1
    work.loc[work["stance"] == "neutral", "signed_confidence"] = 0

    ideos = sorted(work["ideology_group"].dropna().unique())
    fig, axes = plt.subplots(1, len(ideos), figsize=(7*len(ideos), 6))
    if len(ideos) == 1:
        axes = [axes]

    for ax, ideo in zip(axes, ideos):
        subdf = work[work["ideology_group"] == ideo]
        hb = ax.hexbin(subdf["transformer_signed"], subdf["signed_confidence"],
                       gridsize=30, cmap="YlGnBu", mincnt=1, alpha=0.8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_title(f"{ideo.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Transformer Sentiment Score (signed)", fontsize=10)
        ax.set_ylabel("Signed Stance Confidence\n(+Pro, 0 Neutral, -Anti)", fontsize=10)
        plt.colorbar(hb, ax=ax, label="Count")

    plt.suptitle("Sentiment vs Stance Confidence (Transformer)",
                 fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "5b_sentiment_stance_transformer.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_subreddit_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """6. Subreddit-Level Heatmap - stance percentages by subreddit."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["subreddit", "stance", "ideology_group"]).copy()
    if work.empty or "subreddit" not in work.columns:
        return ""

    counts = work.groupby(["subreddit", "stance"]).size().reset_index(name="n")
    totals = counts.groupby("subreddit")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals * 100

    # Ensure all stance columns exist
    for st in ["pro", "anti", "neutral"]:
        if st not in counts["stance"].unique():
            counts = pd.concat([counts, pd.DataFrame([{"subreddit": sr, "stance": st, "n": 0, "prop": 0.0}
                                                      for sr in counts["subreddit"].unique()])], ignore_index=True)

    pivot = counts.pivot_table(index="subreddit", columns="stance", values="prop", fill_value=0)

    ideo_map = work[["subreddit", "ideology_group"]].drop_duplicates().set_index("subreddit")["ideology_group"]
    pivot["_ideology"] = pivot.index.map(ideo_map)
    # Sort: ideology (liberal first), then higher Anti %
    if "anti" not in pivot.columns:
        pivot["anti"] = 0.0
    pivot = pivot.sort_values(["_ideology", "anti"], ascending=[True, False]).drop(columns=["_ideology"])

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot)*0.3)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", center=33.3,
                cbar_kws={"label": "Percentage"}, ax=ax, linewidths=0.5)

    ax.set_title("Stance Distribution by Subreddit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Stance", fontsize=11)
    ax.set_ylabel("Subreddit", fontsize=11)

    out = os.path.join(outdir, "6_subreddit_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_engagement_analysis(df: pd.DataFrame, outdir: str) -> str:
    """7. Score/Engagement Analysis — safely skip if no score/ups column present."""
    os.makedirs(outdir, exist_ok=True)

    # If the dataframe has no 'score', return early (your CSVs don't include it)
    if "score" not in df.columns:
        print("[INFO] Skipping engagement analysis: no 'score' (or ups) column found.")
        return ""

    work = df.dropna(subset=["score", "ideology_group", "stance"]).copy()
    if work.empty:
        print("[INFO] Skipping engagement analysis: no rows with score+ideology+stance.")
        return ""

    # Log-transform (shift by +2 to keep non-positive scores safe)
    work["log_score"] = np.log10(work["score"] + 2)

    ideos = sorted(work["ideology_group"].dropna().unique())
    if len(ideos) == 0:
        print("[INFO] Skipping engagement analysis: ideology_group empty after NA drop.")
        return ""

    fig, axes = plt.subplots(1, len(ideos), figsize=(7*len(ideos), 6))
    if len(ideos) == 1:
        axes = [axes]

    for ax, ideo in zip(axes, ideos):
        subdf = work[work["ideology_group"] == ideo]
        if subdf.empty:
            continue
        sns.boxplot(data=subdf, x="stance", y="log_score", ax=ax,
                    palette=STANCE_COLORS, order=["pro", "anti", "neutral"])
        ax.set_title(f"{ideo.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Stance", fontsize=10)
        ax.set_ylabel("Log10(Score + 2)", fontsize=10)

    plt.suptitle("Engagement (Reddit Scores) by Stance and Ideology",
                 fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "7_engagement_analysis.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

# -------------------------
# QUALITY CONTROL PLOTS
# -------------------------

def plot_classification_diagnostics(df: pd.DataFrame, outdir: str) -> str:
    """8. Classification Diagnostics - NLI score scatter and threshold analysis."""
    os.makedirs(outdir, exist_ok=True)

    if "nli_pro" in df.columns and "nli_anti" in df.columns:
        work = df.dropna(subset=["nli_pro", "nli_anti", "stance"]).copy()
    else:
        # Fallback: approximate from confidence
        work = df.dropna(subset=["confidence", "stance"]).copy()
        work["nli_pro"] = np.where(work["stance"] == "pro", work["confidence"], 1 - work["confidence"])
        work["nli_anti"] = np.where(work["stance"] == "anti", work["confidence"], 1 - work["confidence"])

    if work.empty:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Scatter Pro vs Anti
    ax1 = axes[0]
    for stance, color in STANCE_COLORS.items():
        subset = work[work["stance"] == stance]
        ax1.scatter(subset["nli_pro"], subset["nli_anti"], alpha=0.3, s=10, c=color, label=stance.capitalize())
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label="Decision boundary")
    ax1.set_xlabel("Pro Score", fontsize=10); ax1.set_ylabel("Anti Score", fontsize=10)
    ax1.set_title("NLI Score Distribution", fontsize=12, fontweight="bold")
    ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

    # Histogram of differences
    ax2 = axes[1]
    work["score_diff"] = work["nli_pro"] - work["nli_anti"]
    for stance, color in STANCE_COLORS.items():
        subset = work[work["stance"] == stance]
        ax2.hist(subset["score_diff"], bins=50, alpha=0.5, color=color, label=stance.capitalize(), density=True)
    ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=2)
    ax2.set_xlabel("Score Difference (Pro - Anti)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Score Difference Distribution", fontsize=12, fontweight="bold")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

    # Confidence by max NLI score
    ax3 = axes[2]
    work["max_score"] = work[["nli_pro", "nli_anti"]].max(axis=1)
    for stance, color in STANCE_COLORS.items():
        subset = work[work["stance"] == stance]
        ax3.scatter(subset["max_score"], subset["confidence"], alpha=0.3, s=10, c=color, label=stance.capitalize())
    ax3.set_xlabel("Max NLI Score", fontsize=10); ax3.set_ylabel("Confidence", fontsize=10)
    ax3.set_title("Confidence vs Max Score", fontsize=12, fontweight="bold")
    ax3.legend(loc="best"); ax3.grid(True, alpha=0.3)

    plt.suptitle("Classification Diagnostics", fontsize=14, fontweight="bold")
    out = os.path.join(outdir, "8_classification_diagnostics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

# -------------------------
# HTML REPORT GENERATION
# -------------------------

def generate_html_report(df: pd.DataFrame, outdir: str, plot_paths: Dict[str, str]) -> str:
    """Generate comprehensive HTML report with plots and plain-language explanations."""

    total_posts = int(len(df))
    ideology_counts = df["ideology_group"].value_counts(dropna=True).to_dict() if "ideology_group" in df.columns else {}
    stance_counts = df["stance"].value_counts(dropna=True).to_dict() if "stance" in df.columns else {}
    subject_counts = df["subject"].value_counts(dropna=True).to_dict() if "subject" in df.columns else {}

    if "datetime" in df.columns and df["datetime"].notna().any():
        date_min = df["datetime"].min().strftime("%b %Y")
        date_max = df["datetime"].max().strftime("%b %Y")
        date_range = f"{date_min} – {date_max}"
    else:
        date_range = "Not available"

    # Helper for <img> blocks
    def img_block(path: str, title: str, caption_html: str) -> str:
        if not path:
            return ""
        fname = os.path.basename(path)
        media = (
            f"<iframe src='{fname}' class='plotly'></iframe>"
            if path.endswith(".html")
            else f"<img src='{fname}' alt='{title}' />"
        )
        return f"""
        <section class="card">
            <h2>{title}</h2>
            <div class="caption">{caption_html}</div>
            {media}
        </section>
        """

    # --- CAPTIONS (plain-language, a bit longer) ---
    cap_stance_by_ideol = (
        "Each panel is an <em>EV subject</em> (product, mandate, policy). "
        "Within each panel, bars are stacked proportions that sum to 100% for each ideology group. "
        "<strong>How to read:</strong> taller green means more <em>Pro</em>; taller red means more <em>Anti</em>; "
        "grey is <em>Neutral</em> (the model could not clearly decide). "
        "<strong>What it could mean:</strong> this shows the balance of stances inside each community type, "
        "not absolute volume. A large grey share suggests comments are ambiguous or off-topic for stance."
    )

    cap_stance_no_neutral = (
        "Same as 1), but <strong>Neutral is removed</strong> so the split between clear <em>Pro</em> (green) "
        "and <em>Anti</em> (red) is easier to see. Bars still sum to 100% within ideology. "
        "<strong>How to read:</strong> compare green vs red heights for each ideology to see which stance dominates "
        "among items where the model was confident. "
        "<strong>Caution:</strong> removing Neutral changes the denominator; this plot highlights contrast but "
        "does not show how many items were filtered out as Neutral."
    )

    cap_temporal = (
        "Lines show how stance proportions change <em>over time</em> within each ideology for the selected subject. "
        "Solid lines with circles are <strong>Liberal</strong>, dashed lines with squares are <strong>Conservative</strong>. "
        "Colors indicate stance (green=Pro, red=Anti, grey=Neutral). "
        "<strong>How to read:</strong> for a given month, move vertically to see the share of a stance within an ideology. "
        "<strong>What it could mean:</strong> spikes or dips often correspond to news events or viral threads. "
        "Smaller sample months may look noisy."
    )

    cap_sankey = (
        "The width of each flow is proportional to the number of items. Left to right: <em>Ideology → Subject → Stance</em>. "
        "<strong>How to read:</strong> start from a group on the left, follow the thickest bands to see which subject "
        "people discuss most and how those discussions break into Pro/Anti/Neutral. "
        "<strong>What it could mean:</strong> reveals which topics tend to produce clearer stances and whether one ideology "
        "feeds disproportionately into Anti or Pro outcomes."
    )

    cap_confidence = (
        "Distributions of the model’s <em>confidence</em> (|Pro−Anti|) by ideology and stance, faceted by subject. "
        "<strong>How to read:</strong> a wider violin at a higher value means more items with strong, unambiguous stance. "
        "Neutral items typically cluster near zero since neither Pro nor Anti clearly won. "
        "<strong>What it could mean:</strong> if one ideology shows consistently higher confidence for a stance, "
        "it indicates clearer language patterns for that stance in that community."
    )

    cap_vader = (
        "Each hex cell shows how many items fall at a given combination of <em>sentiment</em> (x-axis) and "
        "<em>signed stance confidence</em> (y-axis: +Pro, 0 Neutral, −Anti). "
        "<strong>How to read:</strong> look for dense areas: far right means positive tone; far left negative tone; "
        "higher means confident Pro; lower means confident Anti. "
        "<strong>What it could mean:</strong> sentiment and stance are related but not identical—"
        "you can have positive tone with Anti stance (e.g., praising a counter-argument) or negative tone with Pro stance "
        "(e.g., criticizing opponents while supporting EVs)."
    )

    cap_transformer = (
        "Same as 5a but using a transformer sentiment classifier (signed by label). "
        "<strong>How to read:</strong> similar to the VADER plot—dense clusters indicate common pairings of sentiment and stance confidence. "
        "<strong>What it could mean:</strong> agreement between 5a and 5b increases trust that the sentiment–stance relationship is real; "
        "disagreement may signal sarcasm or domain vocabulary."
    )

    cap_heatmap = (
        "Rows are subreddits; columns are stance percentages. "
        "<strong>How to read:</strong> each row sums to ~100% across stances. "
        "Green cells mean higher Pro share, red higher Anti, grey higher Neutral. "
        "<strong>What it could mean:</strong> communities with unusually high Anti (or Pro) percentages stand out. "
        "This does not indicate statistical significance—just relative composition."
    )

    cap_engagement = (
        "If available, shows how Reddit score (log scale) varies by stance within each ideology. "
        "<strong>How to read:</strong> compare medians and spread of boxes; higher medians suggest more upvotes on average. "
        "<strong>What it could mean:</strong> a stance that earns higher scores may align better with community expectations, "
        "or reflect topic salience at the time."
    )

    cap_diag = (
        "Quality checks for the stance classifier. Left: scatter of NLI Pro vs Anti scores; the diagonal is the decision boundary. "
        "Middle: distribution of (Pro−Anti); right: how overall confidence relates to the max NLI score. "
        "<strong>How to read:</strong> many points near the diagonal imply harder cases; clear clusters away from it imply easier calls. "
        "<strong>What it could mean:</strong> helps assess whether the threshold is reasonable and whether Neutral volume is expected."
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EV Reddit Stance Analysis Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6; max-width: 1100px; margin: 0 auto; padding: 24px; background: #fafafa; color: #222;
  }}
  .header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;
  }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; margin-top: 8px; }}
  .chip {{
    background: #fff;
    color: #222;                   /* ensure readable text inside chips */
    border: 1px solid rgba(0,0,0,.06);
    border-radius: 10px;
    padding: 8px 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,.06);
  }}
  .chip strong {{ color: #222; }}
  .card {{
    background: #fff; border-radius: 12px; padding: 16px; margin: 18px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
  }}
  h1 {{ margin: 0 0 8px 0; }}
  h2 {{ margin: 0 0 6px 0; }}
  .legend {{
    display:flex; gap:10px; margin: 8px 0 0 0;
  }}
  .legend .swatch {{ width:14px; height:14px; border-radius:3px; display:inline-block; vertical-align:middle; margin-right:6px; }}
  .caption {{ color:#444; margin: 8px 0 14px 0; }}
  img {{ width: 100%; height: auto; border-radius: 8px; }}
  .plotly {{ width: 100%; height: 620px; border: none; border-radius: 8px; background: #fff; }}
</style>
</head>
<body>
  <div class="header">
    <h1>EV Reddit Stance Analysis</h1>
    <div class="meta">
      <div class="chip"><strong>Date range:</strong> {date_range}</div>
      <div class="chip"><strong>Total items:</strong> {total_posts}</div>
      <div class="chip"><strong>Subjects (counts):</strong> {subject_counts}</div>
      <div class="chip"><strong>Stances (counts):</strong> {stance_counts}</div>
    </div>
    <div class="legend">
      <span><span class="swatch" style="background:{STANCE_COLORS['pro']}"></span>Pro</span>
      <span><span class="swatch" style="background:{STANCE_COLORS['anti']}"></span>Anti</span>
      <span><span class="swatch" style="background:{STANCE_COLORS['neutral']}"></span>Neutral</span>
    </div>
    <p style="margin-top:8px;">
      <em>Notes:</em> Stance labels are assigned with a zero-shot MNLI classifier. 
      <strong>Neutral</strong> indicates the model could not confidently support either Pro or Anti hypotheses for the primary subject 
      (i.e., unclear/ambiguous stance by the lightweight BERT model; confidence is |Pro−Anti| after thresholding).
    </p>
  </div>

  {img_block(plot_paths.get("stance_by_ideology",""),
    "1) Stance distribution by ideology (per subject)", cap_stance_by_ideol)}

  {img_block(plot_paths.get("stance_by_ideology_no_neutral",""),
    "1b) Stance distribution by ideology (no Neutral)", cap_stance_no_neutral)}

  {img_block(plot_paths.get("temporal",""),
    "2) Temporal evolution of stances", cap_temporal)}
  
  {img_block(plot_paths.get("sankey",""),
    "3) Discussion flow: Ideology → Subject → Stance", cap_sankey)}

  {img_block(plot_paths.get("confidence",""),
    "4) Confidence distributions", cap_confidence)}

  {img_block(plot_paths.get("vader",""),
    "5a) Sentiment vs stance confidence (VADER)", cap_vader)}

  {img_block(plot_paths.get("transformer",""),
    "5b) Sentiment vs stance confidence (Transformer)", cap_transformer)}

  {img_block(plot_paths.get("heatmap",""),
    "6) Subreddit-level stance distribution", cap_heatmap)}

  {img_block(plot_paths.get("engagement",""),
    "7) Engagement by stance", cap_engagement)}

  {img_block(plot_paths.get("diagnostics",""),
    "8) Classification diagnostics", cap_diag)}

  <div class="card" style="margin-top:22px;">
    <h2>Methodology snapshot</h2>
    <p>
      Data originate from Pushshift Reddit dumps (submissions and comments), filtered by EV-related keyword families 
      (product, mandate, policy) plus negative filters to remove unrelated senses (e.g., electron-volt).
      Subject scores combine raw keyword hits via a saturating transform and the highest scoring subject becomes “primary.”
      Stance is inferred with a zero-shot MNLI model comparing Pro vs Anti hypotheses for that primary subject;
      when neither is confidently supported, the stance is <strong>Neutral</strong>. Sentiment uses VADER and (optionally) a transformer classifier.
    </p>
  </div>
</body>
</html>
"""
    out_html = os.path.join(outdir, "EV_Reddit_Stance_Report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
    """Generate comprehensive HTML report with plots and explanations."""

    total_posts = int(len(df))
    ideology_counts = df["ideology_group"].value_counts(dropna=True).to_dict()
    stance_counts = df["stance"].value_counts(dropna=True).to_dict()
    subject_counts = df["subject"].value_counts(dropna=True).to_dict()

    if "datetime" in df.columns and df["datetime"].notna().any():
        date_min = df["datetime"].min().strftime("%b %Y")
        date_max = df["datetime"].max().strftime("%b %Y")
        date_range = f"{date_min} – {date_max}"
    else:
        date_range = "Not available"

    # Helper for <img> blocks
    def img_block(path: str, title: str, caption: str) -> str:
        if not path:
            return ""
        fname = os.path.basename(path)
        return f"""
        <section class="card">
            <h2>{title}</h2>
            <p class="caption">{caption}</p>
            {"<iframe src='"+fname+"' class='plotly'></iframe>" if path.endswith(".html") else f"<img src='{fname}' alt='{title}' />"}
        </section>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EV Reddit Stance Analysis Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6; max-width: 1100px; margin: 0 auto; padding: 24px; background: #fafafa; color: #222;
  }}
  .header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;
  }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap: 12px; }}
  .chip {{
    background: #fff;
    color: #222;
    border: 1px solid rgba(0,0,0,.06);
    border-radius: 10px;
    padding: 8px 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,.06);
  }}
  .card {{
    background: #fff; border-radius: 12px; padding: 16px; margin: 18px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
  }}
  h1 {{ margin: 0 0 8px 0; }}
  h2 {{ margin: 0 0 6px 0; }}
  .legend {{
    display:flex; gap:10px; margin: 8px 0 0 0;
  }}
  .legend .swatch {{ width:14px; height:14px; border-radius:3px; display:inline-block; vertical-align:middle; margin-right:6px; }}
  .caption {{ color:#555; margin: 8px 0 14px 0; }}
  img {{ width: 100%; height: auto; border-radius: 8px; }}
  .plotly {{ width: 100%; height: 620px; border: none; border-radius: 8px; background: #fff; }}
</style>
</head>
<body>
  <div class="header">
    <h1>EV Reddit Stance Analysis</h1>
    <div class="meta">
      <div class="chip"><strong>Date range:</strong> {date_range}</div>
      <div class="chip"><strong>Total items:</strong> {total_posts}</div>
      <div class="chip"><strong>Subjects (counts):</strong> {subject_counts}</div>
      <div class="chip"><strong>Stances (counts):</strong> {stance_counts}</div>
    </div>
    <div class="legend">
      <span><span class="swatch" style="background:{STANCE_COLORS['pro']}"></span>Pro</span>
      <span><span class="swatch" style="background:{STANCE_COLORS['anti']}"></span>Anti</span>
      <span><span class="swatch" style="background:{STANCE_COLORS['neutral']}"></span>Neutral</span>
    </div>
    <p style="margin-top:8px;">
      <em>Notes:</em> Stance labels are assigned with a zero-shot MNLI classifier. 
      <strong>Neutral</strong> indicates the model could not confidently support either pro or anti hypotheses for the primary subject 
      (i.e., unclear/ambiguous stance by the lightweight BERT model; confidence is |pro−anti| after thresholding).
    </p>
  </div>

  {img_block(plot_paths.get("stance_by_ideology",""), "1a) Stance distribution by ideology (per subject)",
    "Stacked proportions of Pro/Anti/Neutral within each ideology for each subject area (product, mandate, policy).")}

  {img_block(plot_paths.get("stance_by_ideology_no_neutral",""),
    "1b) Stance distribution by ideology (no Neutral)",
    "Neutral removed so Pro vs Anti contrast is clearer within each subject.")}

  {img_block(plot_paths.get("temporal",""), "2) Temporal evolution of stances",
    "Monthly proportions within ideology over time. Solid line: liberal; dashed line: conservative.")}
  
  {img_block(plot_paths.get("sankey",""), "3) Discussion flow: Ideology → Subject → Stance",
    "How posts flow from ideological groups to primary subject and final stance labels.")}

  {img_block(plot_paths.get("confidence",""), "4) Confidence distributions",
    "Model confidence by ideology and stance (higher means a clearer Pro vs Anti separation). Neutral items commonly show lower confidence.")}

  {img_block(plot_paths.get("vader",""), "5a) Sentiment vs stance confidence (VADER)",
    "Signed stance confidence (+Pro, 0 Neutral, −Anti) vs VADER polarity. Sentiment is not the same as stance but can correlate in aggregates.")}

  {img_block(plot_paths.get("transformer",""), "5b) Sentiment vs stance confidence (Transformer)",
    "Signed stance confidence vs transformer sentiment scores (positive vs negative).")}

  {img_block(plot_paths.get("heatmap",""), "6) Subreddit-level stance distribution",
    "Percentage of stances by subreddit. Helpful for spotting communities with comparatively higher Anti or Pro shares.")}

  {img_block(plot_paths.get("engagement",""), "7) Engagement by stance",
    "Log-scaled Reddit score distributions by stance within each ideology. Differences here indicate which stances get more upvotes.")}

  {img_block(plot_paths.get("diagnostics",""), "8) Classification diagnostics",
    "Scatter of subject-specific NLI Pro vs Anti scores, their differences, and relation to overall confidence (useful QC for thresholds).")}

  <div class="card" style="margin-top:22px;">
    <h2>Methodology snapshot</h2>
    <p>
      Data originate from Pushshift Reddit dumps (submissions and comments), filtered by EV-related keyword families 
      (product, mandate, policy) plus negative filters to remove unrelated senses (e.g., electron-volt).
      Subject scores combine raw keyword hits via a saturating transform and the highest scoring subject becomes “primary.”
      Stance is inferred with a zero-shot MNLI model comparing Pro vs Anti hypotheses for that primary subject;
      when neither is confidently supported, the stance is <strong>Neutral</strong>. Sentiment uses VADER and (optionally) a transformer classifier.
    </p>
  </div>
</body>
</html>
"""
    # Write HTML next to plots
    out_html = os.path.join(outdir, "EV_Reddit_Stance_Report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# -------------------------
# DRIVER
# -------------------------



def main():
    ap = argparse.ArgumentParser(description="EV Reddit opinions: analysis and HTML report")
    ap.add_argument("--stance_csv", required=True)
    ap.add_argument("--sentiment_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--confidence_min", type=float, default=0.0,
                    help="Optional filter: keep rows with confidence >= this value")
    ap.add_argument(
    "--export_samples",
    action="store_true",
    help="If set, save merged stance+sentiment CSV and per-final_category samples under the nearest 'results' folder (or out_dir if not found)."
)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, _stance_raw, _sent_raw = load_frames(args.stance_csv, args.sentiment_csv)

    # Optional confidence filter
    if args.confidence_min and "confidence" in df.columns:
        df = df[df["confidence"].fillna(0) >= float(args.confidence_min)]

    # Produce plots
    paths = {}
    paths["stance_by_ideology"] = plot_stance_distribution_by_ideology(df, args.out_dir)
    paths["stance_by_ideology_no_neutral"] = plot_stance_distribution_by_ideology_no_neutral(df, args.out_dir)
    paths["temporal"]           = plot_temporal_evolution(df, args.out_dir)
    paths["sankey"]             = plot_sankey_flow(df, args.out_dir)
    paths["confidence"]         = plot_confidence_distributions(df, args.out_dir)
    paths["vader"]              = plot_sentiment_stance_vader(df, args.out_dir)
    paths["transformer"]        = plot_sentiment_stance_transformer(df, args.out_dir)
    paths["heatmap"]            = plot_subreddit_heatmap(df, args.out_dir)
    paths["engagement"]         = plot_engagement_analysis(df, args.out_dir)
    paths["diagnostics"]        = plot_classification_diagnostics(df, args.out_dir)

    # >>> Optional export (merged CSV + per-category samples) <<<
    if args.export_samples:
        export_dir = export_merged_and_samples(df, args.out_dir)
        print(f"[OK] Exports written under: {export_dir}")

    # Copy plot files into the out_dir root (already saved there), and build HTML
    report = generate_html_report(df, args.out_dir, paths)
    print(f"[OK] Report written to: {report}")



def _resolve_results_root(out_dir: str) -> Path:
    """
    Find the nearest ancestor folder named 'results' (case-insensitive).
    If none, return out_dir itself.
    """
    p = Path(out_dir).resolve()
    for q in [p] + list(p.parents):
        if q.name.lower() == "results":
            return q
    return p

def export_merged_and_samples(df: pd.DataFrame, out_dir: str) -> Path:
    """
    Save merged stance+sentiment CSV and per-final_category samples (<=30 rows) into:
        <results_root>/exports_ev_report/
            merged_stance_sentiment.csv
            samples_by_final_category/<safe_name(category)>.csv
    Returns the export directory path.
    """
    root = _resolve_results_root(out_dir)
    export_dir = root / "exports_ev_report"
    samples_dir = export_dir / "samples_by_final_category"
    export_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Full merged CSV
    merged_csv_path = export_dir / "merged_stance_sentiment.csv"
    df.to_csv(merged_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Merged CSV written to: {merged_csv_path}")

    # Per-category samples
    if "final_category" in df.columns:
        for cat, g in df.groupby("final_category", dropna=False):
            if pd.isna(cat):
                continue
            sample = g.sample(n=min(30, len(g)), random_state=42, replace=False)
            out_path = samples_dir / f"{safe_name(str(cat))}.csv"
            sample.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] Category samples written under: {samples_dir}")
    else:
        print("[WARN] 'final_category' not found; skipping per-category samples.")

    return export_dir


if __name__ == "__main__":
    main()
