"""Public API helpers for running stance labelling and sentiment scoring."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from .fuse_label import LabelDecision, LabelFuser
from .nli_zeroshot import NliScore, ZeroShotScorer, SUBJECTS
from .paths import ensure_parent
from .sentiment import SentimentAnalyzer
from .subjects import SubjectScorer
from .utils import load_yaml
from .weak_rules import WeakRuleScore, WeakRuleScorer

LOGGER = logging.getLogger("evrepo.api")


def _load_inputs(parquet_dir: Path, limit: int | None) -> pl.DataFrame:
    scan = pl.scan_parquet(str(parquet_dir / "**" / "*.parquet"), glob=True)
    scan = scan.select(["id", "created_utc", "subreddit", "ideology_group", "is_submission", "text"])
    if limit:
        scan = scan.head(limit)
    return scan.collect()


def _deduplicate_texts(texts: List[str]) -> Tuple[List[str], List[int]]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    refs: List[int] = []
    for text in texts:
        key = text or ""
        idx = seen.get(key)
        if idx is None:
            idx = len(unique)
            unique.append(key)
            seen[key] = idx
        refs.append(idx)
    return unique, refs


def _zero_weak_scores() -> Dict[str, WeakRuleScore]:
    zero = WeakRuleScore(pro=0.0, anti=0.0, pro_hits=0, anti_hits=0)
    return {subject: zero for subject in SUBJECTS}


def _verify_against_single(
    scorer: ZeroShotScorer,
    texts: List[str],
    batch_scores: Dict[str, List[NliScore]],
    max_rows: int = 3,
) -> Dict[str, float]:
    rows = min(len(texts), max_rows)
    mae: Dict[str, float] = {}
    if rows == 0:
        return mae
    mae_store: Dict[str, List[float]] = {f"{subject}_pro": [] for subject in SUBJECTS}
    mae_store.update({f"{subject}_anti": [] for subject in SUBJECTS})
    match_count: Dict[str, int] = {subject: 0 for subject in SUBJECTS}
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
                match_count[subject] += 1
    for key, values in mae_store.items():
        mae[key] = sum(values) / len(values) if values else 0.0
    for subject in SUBJECTS:
        mae[f"{subject}_match_count"] = match_count[subject]
    mae["verify_rows"] = rows
    return mae


def run_stance_label(
    parquet_dir: str | Path,
    out_csv: str | Path,
    *,
    limit: int | None = None,
    log_level: str = "INFO",
    debug: bool = False,
    use_weak_rules: bool = False,
    rules_mode: str = "simple",
    fast_model: bool = True,
    backend: str = "torch",
    batch_size: int = 32,
    verify: bool = False,
    resume: bool = True,
    overwrite: bool = False,
) -> Dict[str, float]:
    """Label stances and write a CSV. Returns timing/row stats."""

    logger = logging.getLogger("evrepo.label_stance")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if debug:
        logger.setLevel(logging.DEBUG)

    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    if backend == "onnx":
        raise NotImplementedError("ONNX backend is TODO")

    keyword_cfg = load_yaml("config/keywords.yaml")
    labeling_cfg = load_yaml("config/labeling.yaml")

    subject_scorer = SubjectScorer(keyword_cfg)
    weak_scorer = WeakRuleScorer(keyword_cfg, mode=rules_mode) if use_weak_rules else None
    nli_scorer = ZeroShotScorer(fast_model=fast_model, backend=backend)
    fuser = LabelFuser(labeling_cfg.get("fusion"), labeling_cfg.get("subject_tie_break_priority", []))

    start_time = time.perf_counter()
    # If output exists, build a set of already-processed IDs for skip/resume.
    out_path = Path(out_csv)
    processed_ids: set[str] = set()
    skipped_count = 0
    if resume and not overwrite and out_path.exists():
        try:
            existing = pl.read_csv(out_path, ignore_errors=True)
            if "id" in existing.columns and existing.height > 0:
                processed_ids = set(existing.get_column("id").cast(pl.Utf8).to_list())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to read existing stance CSV %s: %s", out_path, exc)

    df = _load_inputs(parquet_path, limit)
    if processed_ids:
        skipped_count = len(processed_ids)
        df = df.filter(~pl.col("id").cast(pl.Utf8).is_in(list(processed_ids)))
    if df.height == 0:
        return {"rows": 0, "elapsed": 0.0, "rows_per_sec": 0.0}

    rows = df.to_dicts()
    texts = [row.get("text") or "" for row in rows]
    unique_texts, back_refs = _deduplicate_texts(texts)
    logger.info("Prepared %d unique texts from %d rows", len(unique_texts), len(texts))

    nli_unique = nli_scorer.score_all(unique_texts, batch_size=batch_size)
    nli_scores: Dict[str, List[NliScore]] = {
        subject: [nli_unique[subject][idx] for idx in back_refs]
        for subject in SUBJECTS
    }

    if debug and verify:
        metrics = _verify_against_single(nli_scorer, texts, nli_scores)
        for key, value in metrics.items():
            logger.debug("verify %s=%s", key, value)

    zero_scores = _zero_weak_scores()
    output_rows: List[Dict[str, object]] = []
    debug_counter = 0

    for i, row in enumerate(rows):
        text_value = texts[i]
        subject_scores = subject_scorer.score(text_value)
        nli_row = {subject: nli_scores[subject][i] for subject in SUBJECTS}
        if use_weak_rules and weak_scorer is not None:
            weak_scores = weak_scorer.score(text_value)
            decision: LabelDecision = fuser.fuse(subject_scores, weak_scores, nli_row)
        else:
            decision: LabelDecision = fuser.decide_nli_only(subject_scores, nli_row)

        if logger.isEnabledFor(logging.DEBUG) and debug_counter < 3:
            logger.debug(
                "Row id=%s subject_raw=%s subject_norm=%s",
                row.get("id"),
                subject_scores.raw,
                subject_scores.normalized,
            )
            if use_weak_rules and weak_scorer is not None:
                logger.debug(
                    "Row id=%s weak_scores=%s",
                    row.get("id"),
                    {k: vars(v) for k, v in weak_scores.items()},
                )
            logger.debug(
                "Row id=%s nli_scores=%s",
                row.get("id"),
                {k: vars(v) for k, v in nli_row.items()},
            )
            logger.debug(
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
            "is_submission": row.get("is_submission"),
            "record_type": "post" if bool(row.get("is_submission")) else "comment",
            "text": text_value,
            "subject_product_raw": subject_scores.raw.get("product"),
            "subject_product_norm": subject_scores.normalized.get("product"),
            "subject_mandate_raw": subject_scores.raw.get("mandate"),
            "subject_mandate_norm": subject_scores.normalized.get("mandate"),
            "subject_policy_raw": subject_scores.raw.get("policy"),
            "subject_policy_norm": subject_scores.normalized.get("policy"),
            "nli_product_pro": nli_row["product"].pro,
            "nli_product_anti": nli_row["product"].anti,
            "nli_mandate_pro": nli_row["mandate"].pro,
            "nli_mandate_anti": nli_row["mandate"].anti,
            "nli_policy_pro": nli_row["policy"].pro,
            "nli_policy_anti": nli_row["policy"].anti,
            "final_subject": decision.subject,
            "final_stance": decision.stance,
            "final_category": decision.category,
            "fused_pro": decision.fused_pro,
            "fused_anti": decision.fused_anti,
            "confidence": decision.confidence,
        }
        if use_weak_rules and weak_scorer is not None:
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
        output_rows.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pl.DataFrame(output_rows)
    # If file exists and not overwriting, merge (dedup by id) to ensure idempotency.
    if overwrite:
        new_df.write_csv(out_path)
    elif out_path.exists():
        try:
            prior = pl.read_csv(out_path, ignore_errors=True)
            if prior.height > 0:
                combined = (
                    pl.concat([prior, new_df], how="vertical", rechunk=True)
                    .unique(subset=["id"], keep="first")
                )
                combined.write_csv(out_path)
            else:
                new_df.write_csv(out_path)
        except Exception:
            # Fallback: overwrite with new only
            new_df.write_csv(out_path)
    else:
        new_df.write_csv(out_path)
    elapsed = time.perf_counter() - start_time
    rows_processed = len(output_rows)
    rows_per_sec = rows_processed / elapsed if elapsed else float("inf")
    logger.info(
        "Wrote %d labelled rows to %s (skipped_existing_ids=%d, overwrite=%s)",
        rows_processed,
        out_path,
        skipped_count,
        overwrite,
    )
    logger.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, rows_per_sec)

    return {"rows": rows_processed, "elapsed": elapsed, "rows_per_sec": rows_per_sec}


def run_sentiment(
    parquet_dir: str | Path,
    out_csv: str | Path,
    *,
    limit: int | None = None,
    log_level: str = "INFO",
    debug: bool = False,
    resume: bool = True,
    overwrite: bool = False,
) -> Dict[str, float]:
    """Compute sentiment scores and write a CSV."""

    logger = logging.getLogger("evrepo.score_sentiment")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if debug:
        logger.setLevel(logging.DEBUG)

    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    start_time = time.perf_counter()
    out_path = Path(out_csv)
    processed_ids: set[str] = set()
    skipped_count = 0
    if resume and not overwrite and out_path.exists():
        try:
            existing = pl.read_csv(out_path, ignore_errors=True)
            if "id" in existing.columns and existing.height > 0:
                processed_ids = set(existing.get_column("id").cast(pl.Utf8).to_list())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to read existing sentiment CSV %s: %s", out_path, exc)

    df = _load_inputs(parquet_path, limit)
    if processed_ids:
        skipped_count = len(processed_ids)
        df = df.filter(~pl.col("id").cast(pl.Utf8).is_in(list(processed_ids)))
    if df.height == 0:
        return {"rows": 0, "elapsed": 0.0, "rows_per_sec": 0.0}

    rows = df.to_dicts()
    analyzer = SentimentAnalyzer()
    results: List[Dict[str, object]] = []
    debug_counter = 0

    for row in rows:
        text_value = row.get("text") or ""
        sentiment = analyzer.score(text_value)
        if logger.isEnabledFor(logging.DEBUG) and debug_counter < 3:
            logger.debug(
                "Row id=%s sentiment vader=%.4f transformer=%s(%.4f)",
                row.get("id"),
                sentiment.vader_compound,
                sentiment.transformer_label,
                sentiment.transformer_score,
            )
            debug_counter += 1
        results.append(
            {
                "id": row.get("id"),
                "created_utc": row.get("created_utc"),
                "subreddit": row.get("subreddit"),
                "ideology_group": row.get("ideology_group"),
                "is_submission": row.get("is_submission"),
                "record_type": "post" if bool(row.get("is_submission")) else "comment",
                "text": text_value,
                "sent_vader_compound": sentiment.vader_compound,
                "sent_transformer_label": sentiment.transformer_label,
                "sent_transformer_score": sentiment.transformer_score,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pl.DataFrame(results)
    if overwrite:
        new_df.write_csv(out_path)
    elif out_path.exists():
        try:
            prior = pl.read_csv(out_path, ignore_errors=True)
            if prior.height > 0:
                combined = (
                    pl.concat([prior, new_df], how="vertical", rechunk=True)
                    .unique(subset=["id"], keep="first")
                )
                combined.write_csv(out_path)
            else:
                new_df.write_csv(out_path)
        except Exception:
            new_df.write_csv(out_path)
    else:
        new_df.write_csv(out_path)
    elapsed = time.perf_counter() - start_time
    rows_processed = len(results)
    rows_per_sec = rows_processed / elapsed if elapsed else float("inf")
    logger.info(
        "Wrote %d sentiment rows to %s (skipped_existing_ids=%d, overwrite=%s)",
        rows_processed,
        out_path,
        skipped_count,
        overwrite,
    )
    logger.info("Completed in %.2f seconds (rows/sec=%.2f)", elapsed, rows_per_sec)

    return {"rows": rows_processed, "elapsed": elapsed, "rows_per_sec": rows_per_sec}
