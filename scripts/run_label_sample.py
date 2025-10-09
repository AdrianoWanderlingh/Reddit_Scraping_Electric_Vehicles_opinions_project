# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Label EV stances from Parquet inputs using keyword subjects and batched MNLI."""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Dict, List

import polars as pl

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evrepo.fuse_label import LabelDecision, LabelFuser
from evrepo.nli_zeroshot import NliScore, ZeroShotScorer
from evrepo.subjects import SUBJECTS, SubjectScorer
from evrepo.utils import load_yaml
from evrepo.weak_rules import WeakRuleScore, WeakRuleScorer

LOGGER = logging.getLogger("evrepo.run_label_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stance labelling on Parquet inputs")
    parser.add_argument("--parquet_dir", default="data/parquet_sample/year=2024", help="Directory containing Parquet partitions")
    parser.add_argument("--out_csv", default="data/stance_labels.csv", help="Destination CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap (applied lazily)")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--use_weak_rules", action="store_true", help="Enable weak-rule cues (off by default)")
    parser.add_argument(
        "--rules_mode",
        choices=["simple", "full"],
        default="simple",
        help="Weak-rule mode when enabled (default simple)",
    )
    parser.add_argument("--verify", action="store_true", help="Compare single vs batched MNLI on up to 3 rows")
    return parser.parse_args()


def load_inputs(parquet_dir: Path, limit: int | None) -> pl.DataFrame:
    scan = pl.scan_parquet(str(parquet_dir / "**" / "*.parquet"), glob=True)
    scan = scan.select(["id", "created_utc", "subreddit", "ideology_group", "text"])
    if limit:
        scan = scan.head(limit)
    return scan.collect()


def deduplicate_texts(texts: List[str]) -> tuple[List[str], List[int]]:
    unique: Dict[str, int] = {}
    unique_texts: List[str] = []
    back_refs: List[int] = []
    for text in texts:
        key = text or ""
        if key not in unique:
            unique[key] = len(unique_texts)
            unique_texts.append(key)
        back_refs.append(unique[key])
    return unique_texts, back_refs


def prepare_nli_scores(
    scorer: ZeroShotScorer,
    texts: List[str],
    back_refs: List[int],
    batch_size: int = 16,
) -> Dict[str, List[NliScore]]:
    unique_texts = texts
    subject_batches = {
        subject: scorer.score_batch(unique_texts, subject, batch_size=batch_size)
        for subject in SUBJECTS
    }
    scattered: Dict[str, List[NliScore]] = {}
    for subject, scores in subject_batches.items():
        scattered[subject] = [scores[idx] for idx in back_refs]
    return scattered


def zeros_weak_scores() -> Dict[str, WeakRuleScore]:
    zero = WeakRuleScore(pro=0.0, anti=0.0, pro_hits=0, anti_hits=0)
    return {subject: zero for subject in SUBJECTS}


def verify_against_single(
    scorer: ZeroShotScorer,
    texts: List[str],
    batch_scores: Dict[str, List[NliScore]],
    max_rows: int = 3,
) -> None:
    rows = min(len(texts), max_rows)
    if rows == 0:
        return
    LOGGER.info("Running lenient verify on %d rows", rows)
    mae_store: Dict[str, List[float]] = {f"{subject}_pro": [] for subject in SUBJECTS}
    mae_store.update({f"{subject}_anti": [] for subject in SUBJECTS})
    label_matches: Dict[str, int] = {subject: 0 for subject in SUBJECTS}
    for idx in range(rows):
        text = texts[idx]
        for subject in SUBJECTS:
            single = scorer.score(text, subject)
            batch = batch_scores[subject][idx]
            mae_store[f"{subject}_pro"].append(abs(single.pro - batch.pro))
            mae_store[f"{subject}_anti"].append(abs(single.anti - batch.anti))
            if math.isclose(single.pro, batch.pro, abs_tol=1e-3) and math.isclose(
                single.anti,
                batch.anti,
                abs_tol=1e-3,
            ):
                label_matches[subject] += 1
    for key, values in mae_store.items():
        if values:
            LOGGER.info("verify mae %s=%.6f", key, sum(values) / len(values))
    for subject in SUBJECTS:
        LOGGER.info("verify matches %s=%d/%d", subject, label_matches[subject], rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    keyword_cfg = load_yaml("config/keywords.yaml")
    labeling_cfg = load_yaml("config/labeling.yaml")

    subject_scorer = SubjectScorer(keyword_cfg)
    weak_scorer = WeakRuleScorer(keyword_cfg, mode=args.rules_mode) if args.use_weak_rules else None
    nli_scorer = ZeroShotScorer()
    fuser = LabelFuser(labeling_cfg.get("fusion"), labeling_cfg.get("subject_tie_break_priority", []))

    start_time = time.perf_counter()
    df = load_inputs(parquet_dir, args.limit)
    if df.height == 0:
        LOGGER.warning("No rows found in %s", parquet_dir)
        elapsed = time.perf_counter() - start_time
        LOGGER.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, 0.0)
        return

    texts = [value or "" for value in df["text"].to_list()]
    unique_texts, back_refs = deduplicate_texts(texts)
    LOGGER.info("Prepared %d unique texts from %d rows", len(unique_texts), len(texts))

    nli_scores = prepare_nli_scores(nli_scorer, unique_texts, back_refs, batch_size=16)

    if args.verify:
        verify_against_single(nli_scorer, texts, nli_scores, max_rows=3)

    results: List[dict] = []
    debug_counter = 0
    zero_scores = zeros_weak_scores()

    for idx, row in enumerate(df.iter_rows(named=True)):
        text_value = texts[idx]
        subject_scores = subject_scorer.score(text_value)
        if args.use_weak_rules and weak_scorer is not None:
            weak_scores = weak_scorer.score(text_value)
        else:
            weak_scores = zero_scores
        nli_row_scores = {subject: nli_scores[subject][idx] for subject in SUBJECTS}
        decision: LabelDecision = fuser.fuse(subject_scores, weak_scores, nli_row_scores)

        if LOGGER.isEnabledFor(logging.DEBUG) and debug_counter < 3:
            LOGGER.debug(
                "Row id=%s subject_raw=%s subject_norm=%s",
                row.get("id"),
                subject_scores.raw,
                subject_scores.normalized,
            )
            if args.use_weak_rules and weak_scorer is not None:
                LOGGER.debug(
                    "Row id=%s weak_scores=%s",
                    row.get("id"),
                    {k: vars(v) for k, v in weak_scores.items()},
                )
            LOGGER.debug(
                "Row id=%s nli_scores=%s",
                row.get("id"),
                {k: vars(v) for k, v in nli_row_scores.items()},
            )
            LOGGER.debug(
                "Row id=%s decision subject=%s stance=%s fused_pro=%.4f fused_anti=%.4f confidence=%.4f",
                row.get("id"),
                decision.subject,
                decision.stance,
                decision.fused_pro,
                decision.fused_anti,
                decision.confidence,
            )
            debug_counter += 1

        record = {
            "id": row.get("id"),
            "created_utc": row.get("created_utc"),
            "subreddit": row.get("subreddit"),
            "ideology_group": row.get("ideology_group"),
            "text": text_value,
            "subject_product_raw": subject_scores.raw.get("product"),
            "subject_product_norm": subject_scores.normalized.get("product"),
            "subject_mandate_raw": subject_scores.raw.get("mandate"),
            "subject_mandate_norm": subject_scores.normalized.get("mandate"),
            "subject_policy_raw": subject_scores.raw.get("policy"),
            "subject_policy_norm": subject_scores.normalized.get("policy"),
            "nli_product_pro": nli_row_scores["product"].pro,
            "nli_product_anti": nli_row_scores["product"].anti,
            "nli_mandate_pro": nli_row_scores["mandate"].pro,
            "nli_mandate_anti": nli_row_scores["mandate"].anti,
            "nli_policy_pro": nli_row_scores["policy"].pro,
            "nli_policy_anti": nli_row_scores["policy"].anti,
            "final_subject": decision.subject,
            "final_stance": decision.stance,
            "final_category": decision.category,
            "fused_pro": decision.fused_pro,
            "fused_anti": decision.fused_anti,
            "confidence": decision.confidence,
        }
        if args.use_weak_rules and weak_scorer is not None:
            record.update(
                {
                    "weak_product_pro": weak_scores["product"].pro,
                    "weak_product_anti": weak_scores["product"].anti,
                    "weak_mandate_pro": weak_scores["mandate"].pro,
                    "weak_mandate_anti": weak_scores["mandate"].anti,
                    "weak_policy_pro": weak_scores["policy"].pro,
                    "weak_policy_anti": weak_scores["policy"].anti,
                }
            )
        results.append(record)

    output = pl.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(out_path)
    elapsed = time.perf_counter() - start_time
    rows = len(results)
    rows_per_sec = rows / elapsed if elapsed else float("inf")
    LOGGER.info("Wrote %d labelled rows to %s", rows, out_path)
    LOGGER.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, rows_per_sec)


if __name__ == "__main__":
    main()
