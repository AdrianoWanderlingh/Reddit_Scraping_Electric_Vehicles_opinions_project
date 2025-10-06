# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Process Parquet samples to assign EV stance labels and emit a CSV for review."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evrepo.fuse_label import LabelFuser
from evrepo.nli_zeroshot import NliScore, ZeroShotScorer
from evrepo.subjects import SUBJECTS, SubjectScorer
from evrepo.utils import load_yaml
from evrepo.weak_rules import WeakRuleScorer
from evrepo.sentiment import SentimentAnalyzer

LOGGER = logging.getLogger("evrepo.run_label_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject scoring and stance fusion on sample data")
    parser.add_argument("--parquet_dir", default="data/parquet_sample/year=2024", help="Root directory containing Parquet partitions to label")
    parser.add_argument("--out_csv", default="data/sample_labels.csv", help="Destination CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick debugging")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def load_inputs(parquet_dir: Path, limit: int | None) -> pl.DataFrame:
    scan = pl.scan_parquet(str(parquet_dir / '**' / '*.parquet'), glob=True)
    if limit:
        scan = scan.head(limit)
    return scan.collect()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    keyword_cfg = load_yaml("config/keywords.yaml")
    labeling_cfg = load_yaml("config/labeling.yaml")

    subject_scorer = SubjectScorer(keyword_cfg)
    weak_scorer = WeakRuleScorer(keyword_cfg)
    nli_scorer = ZeroShotScorer()
    sentiment_analyzer = SentimentAnalyzer()
    fuser = LabelFuser(labeling_cfg.get("fusion"), labeling_cfg.get("subject_tie_break_priority", []))

    df = load_inputs(parquet_dir, args.limit)
    if df.height == 0:
        LOGGER.warning("No rows found in %s", parquet_dir)
        return

    results: list[dict] = []
    for row in df.iter_rows(named=True):
        text = row.get("text") or ""
        subject_scores = subject_scorer.score(text)
        weak_scores = weak_scorer.score(text)
        nli_scores: Dict[str, NliScore] = {subject: nli_scorer.score(text, subject) for subject in SUBJECTS}
        decision = fuser.fuse(subject_scores, weak_scores, nli_scores)
        sentiment = sentiment_analyzer.score(text)

        record = {
            "id": row.get("id"),
            "created_utc": row.get("created_utc"),
            "subreddit": row.get("subreddit"),
            "ideology_group": row.get("ideology_group"),
            "text": text,
            "subject_product_raw": subject_scores.raw.get("product"),
            "subject_product_norm": subject_scores.normalized.get("product"),
            "subject_mandate_raw": subject_scores.raw.get("mandate"),
            "subject_mandate_norm": subject_scores.normalized.get("mandate"),
            "subject_policy_raw": subject_scores.raw.get("policy"),
            "subject_policy_norm": subject_scores.normalized.get("policy"),
            "weak_product_pro": weak_scores["product"].pro,
            "weak_product_anti": weak_scores["product"].anti,
            "weak_mandate_pro": weak_scores["mandate"].pro,
            "weak_mandate_anti": weak_scores["mandate"].anti,
            "weak_policy_pro": weak_scores["policy"].pro,
            "weak_policy_anti": weak_scores["policy"].anti,
            "nli_product_pro": nli_scores["product"].pro,
            "nli_product_anti": nli_scores["product"].anti,
            "nli_mandate_pro": nli_scores["mandate"].pro,
            "nli_mandate_anti": nli_scores["mandate"].anti,
            "nli_policy_pro": nli_scores["policy"].pro,
            "nli_policy_anti": nli_scores["policy"].anti,
            "sent_vader_compound": sentiment.vader_compound,
            "sent_transformer_label": sentiment.transformer_label,
            "sent_transformer_score": sentiment.transformer_score,
            "final_subject": decision.subject,
            "final_stance": decision.stance,
            "final_category": decision.category,
            "fused_pro": decision.fused_pro,
            "fused_anti": decision.fused_anti,
            "confidence": decision.confidence,
        }
        results.append(record)

    output = pl.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(out_path)
    LOGGER.info("Wrote %d labelled rows to %s", len(results), out_path)


if __name__ == "__main__":
    main()
