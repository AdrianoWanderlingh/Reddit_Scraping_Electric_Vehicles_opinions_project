#!/usr/bin/env python
from __future__ import annotations

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
from matplotlib import cm
from matplotlib.patches import Patch

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:  # pragma: no cover - fallback for lightweight environments
    sns = None
    HAVE_SEABORN = False
    print("[WARN] seaborn not available; using matplotlib fallbacks. Install via 'pip install seaborn'.")

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
if HAVE_SEABORN:
    sns.set_palette("Set2")
else:
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cm.get_cmap("Set2").colors)

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
    """Load and merge stance + sentiment CSVs with robust parsing."""
    stance = pd.read_csv(stance_csv)
    sent = pd.read_csv(sentiment_csv)

    # Normalize IDs
    stance = stance.rename(columns={require_id_column(stance): "id"})
    sent   = sent.rename(columns={require_id_column(sent): "id"})

    # Normalize ideology in both
    for df in (stance, sent):
        if "ideology_group" in df.columns:
            df["ideology_group"] = df["ideology_group"].astype(str).str.strip().str.lower()

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

    # Derive subject & stance — prefer explicit columns if present
    if "final_subject" in merged.columns:
        merged["subject"] = merged["final_subject"].astype(str).str.strip().str.lower()
    else:
        # Fallback: try to parse from final_category like "Pro-EV (product)"
        if "final_category" in merged.columns:
            merged["subject"] = merged["final_category"].str.extract(r"\((product|mandate|policy)\)", expand=False).str.lower()
        else:
            merged["subject"] = np.nan

    if "final_stance" in merged.columns:
        merged["stance"] = merged["final_stance"].astype(str).str.strip().str.lower()
    else:
        # Fallback: parse from strings like "Pro-EV", "Against-EV", "Neutral-EV"
        if "final_category" in merged.columns:
            merged["stance"] = (
                merged["final_category"]
                .str.extract(r"^(Pro|Against|Neutral)", expand=False)
                .str.lower()
                .replace({"pro": "pro", "against": "anti", "neutral": "neutral"})
            )
        else:
            merged["stance"] = np.nan

    # Parse created_utc for temporal analysis
    created_col = find_column(merged, POSSIBLE_CREATED_COLUMNS)
    if created_col:
        merged["created_utc"] = pd.to_numeric(merged[created_col], errors="coerce")
        merged["datetime"] = pd.to_datetime(merged["created_utc"], unit="s", errors="coerce")
        merged["year_month"] = merged["datetime"].dt.to_period("M")

    # Score column for engagement analysis
    score_col = find_column(merged, POSSIBLE_SCORE_COLUMNS)
    if score_col and score_col != "score":
        merged = merged.rename(columns={score_col: "score"})
    if "score" in merged.columns:
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")

    # Subreddit column
    sub_col = find_column(merged, POSSIBLE_SUBREDDIT_COLUMNS)
    if sub_col and sub_col != "subreddit":
        merged = merged.rename(columns={sub_col: "subreddit"})

    # Confidence to numeric
    if "confidence" in merged.columns:
        merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")

    # Derive subject-specific NLI scores into generic columns for diagnostics
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
                merged.loc[mask, "nli_pro"] = pd.to_numeric(merged.loc[mask, pro_col], errors="coerce")
            if anti_col in merged.columns:
                merged.loc[mask, "nli_anti"] = pd.to_numeric(merged.loc[mask, anti_col], errors="coerce")

    # Keep only valid subjects/stances
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
STANCE_ORDER = ["pro", "anti", "neutral"]


def _violinplot_fallback(ax: plt.Axes, data: pd.DataFrame, *, x: str, y: str, hue: str) -> None:
    """Draw a basic violin plot using matplotlib when seaborn is unavailable."""
    categories = [c for c in data[x].dropna().unique()]
    stances = [s for s in STANCE_ORDER if s in data[hue].dropna().unique()]
    if not categories or not stances:
        return

    width = 0.8 / max(len(stances), 1)
    centers = np.arange(len(categories), dtype=float)

    for idx, cat in enumerate(categories):
        for jdx, stance in enumerate(stances):
            subset = data[(data[x] == cat) & (data[hue] == stance)][y].dropna()
            if subset.empty:
                continue
            offset = (jdx - (len(stances) - 1) / 2.0) * width
            position = centers[idx] + offset
            parts = ax.violinplot(
                subset.values,
                positions=[position],
                widths=width * 0.9,
                showextrema=False,
                showmeans=False,
                showmedians=True,
            )
            for body in parts["bodies"]:
                body.set_facecolor(STANCE_COLORS.get(stance, "#888888"))
                body.set_edgecolor("black")
                body.set_alpha(0.65)
            if parts.get("cmedians") is not None:
                parts["cmedians"].set_color("black")

    ax.set_xticks(centers)
    ax.set_xticklabels([str(cat).capitalize() for cat in categories])

    handles = [
        Patch(facecolor=STANCE_COLORS.get(stance, "#888888"), edgecolor="black", label=stance.capitalize())
        for stance in stances
    ]
    if handles:
        ax.legend(handles=handles, title="Stance", loc="lower right")


def _heatmap_fallback(ax: plt.Axes, pivot: pd.DataFrame) -> None:
    """Render a heatmap without seaborn using matplotlib primitives."""
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=np.nanmin(data), vmax=np.nanmax(data))
    plt.colorbar(im, ax=ax, label="Percentage")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(col).capitalize() for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(idx) for idx in pivot.index])


def _boxplot_fallback(ax: plt.Axes, data: pd.DataFrame, *, x: str, y: str) -> None:
    """Simple grouped boxplot without seaborn."""
    stances = [s for s in STANCE_ORDER if s in data[x].dropna().unique()]
    if not stances:
        return

    values = [data[data[x] == stance][y].dropna().values for stance in stances]
    positions = np.arange(1, len(stances) + 1)
    bp = ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True)
    for patch, stance in zip(bp["boxes"], stances):
        patch.set_facecolor(STANCE_COLORS.get(stance, "#888888"))
        patch.set_edgecolor("black")
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps", "medians", "fliers"):
        for artist in bp[element]:
            artist.set(color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([s.capitalize() for s in stances])

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

def plot_temporal_evolution(df: pd.DataFrame, outdir: str) -> str:
    """2. Temporal Evolution - multi-line time series by ideology × stance."""
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

        for ideo in sorted(monthly["ideology_group"].unique()):
            for stance in ["pro", "anti", "neutral"]:
                subset = monthly[(monthly["ideology_group"] == ideo) & (monthly["stance"] == stance)]
                if subset.empty:
                    continue
                linestyle = "-" if ideo == "liberal" else "--"
                color = STANCE_COLORS.get(stance, "#777777")
                label = f"{ideo.capitalize()} {stance.capitalize()}"
                ax.plot(subset["date"], subset["prop"],
                        linestyle=linestyle, color=color, linewidth=2,
                        marker="o", markersize=3, label=label, alpha=0.8)

        ax.set_title(f"{subj.capitalize()} Stances Over Time",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Proportion within Ideology", fontsize=10)
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Temporal Evolution of Stances by Ideology",
                 fontsize=14, fontweight="bold")
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
        if HAVE_SEABORN:
            sns.violinplot(
                data=subdf,
                x="ideology_group",
                y="confidence",
                hue="stance",
                split=False,
                ax=ax,
                inner="quartile",
                palette=STANCE_COLORS,
            )
            ax.legend(title="Stance", loc="lower right")
        else:
            _violinplot_fallback(ax, subdf, x="ideology_group", y="confidence", hue="stance")
        ax.set_title(f"{subj.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ideology Group", fontsize=10)
        ax.set_ylabel("Stance Confidence", fontsize=10)

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
    if HAVE_SEABORN:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            center=33.3,
            cbar_kws={"label": "Percentage"},
            ax=ax,
            linewidths=0.5,
        )
    else:
        _heatmap_fallback(ax, pivot)

    ax.set_title("Stance Distribution by Subreddit", fontsize=14, fontweight="bold")
    ax.set_xlabel("Stance", fontsize=11)
    ax.set_ylabel("Subreddit", fontsize=11)

    out = os.path.join(outdir, "6_subreddit_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_engagement_analysis(df: pd.DataFrame, outdir: str) -> str:
    """7. Score/Engagement Analysis - box plots of Reddit scores."""
    os.makedirs(outdir, exist_ok=True)
    work = df.dropna(subset=["score", "ideology_group", "stance"]).copy()
    if work.empty or "score" not in work.columns:
        return ""

    work["log_score"] = np.log10(work["score"] + 2)

    ideos = sorted(work["ideology_group"].dropna().unique())
    fig, axes = plt.subplots(1, len(ideos), figsize=(7*len(ideos), 6))
    if len(ideos) == 1:
        axes = [axes]

    for ax, ideo in zip(axes, ideos):
        subdf = work[work["ideology_group"] == ideo]
        if HAVE_SEABORN:
            sns.boxplot(data=subdf, x="stance", y="log_score", ax=ax, palette=STANCE_COLORS)
        else:
            _boxplot_fallback(ax, subdf, x="stance", y="log_score")
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
  .chip {{ background: #fff; border-radius: 10px; padding: 8px 12px; box-shadow: 0 2px 6px rgba(0,0,0,.06); }}
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

  {img_block(plot_paths.get("stance_by_ideology",""), "1) Stance distribution by ideology (per subject)",
    "Stacked proportions of Pro/Anti/Neutral within each ideology for each subject area (product, mandate, policy).")}

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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, _stance_raw, _sent_raw = load_frames(args.stance_csv, args.sentiment_csv)

    # Optional confidence filter
    if args.confidence_min and "confidence" in df.columns:
        df = df[df["confidence"].fillna(0) >= float(args.confidence_min)]

    # Produce plots
    paths = {}
    paths["stance_by_ideology"] = plot_stance_distribution_by_ideology(df, args.out_dir)
    paths["temporal"]           = plot_temporal_evolution(df, args.out_dir)
    paths["sankey"]             = plot_sankey_flow(df, args.out_dir)
    paths["confidence"]         = plot_confidence_distributions(df, args.out_dir)
    paths["vader"]              = plot_sentiment_stance_vader(df, args.out_dir)
    paths["transformer"]        = plot_sentiment_stance_transformer(df, args.out_dir)
    paths["heatmap"]            = plot_subreddit_heatmap(df, args.out_dir)
    paths["engagement"]         = plot_engagement_analysis(df, args.out_dir)
    paths["diagnostics"]        = plot_classification_diagnostics(df, args.out_dir)

    # Copy plot files into the out_dir root (already saved there), and build HTML
    report = generate_html_report(df, args.out_dir, paths)
    print(f"[OK] Report written to: {report}")

if __name__ == "__main__":
    main()
