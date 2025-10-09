# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Zero-shot MNLI scoring using HuggingFace transformers."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - external dependency missing
    pipeline = None  # type: ignore

LOGGER = logging.getLogger("evrepo.nli")

HYPOTHESES: Dict[str, Dict[str, str]] = {
    "product": {
        "pro": "This text supports electric cars as a product.",
        "anti": "This text opposes electric cars as a product.",
    },
    "mandate": {
        "pro": "This text supports mandates or requirements for electric vehicles.",
        "anti": "This text opposes mandates or requirements for electric vehicles.",
    },
    "policy": {
        "pro": "This text supports EV-related policies such as subsidies or regulations (excluding mandates).",
        "anti": "This text opposes EV-related policies such as subsidies or regulations (excluding mandates).",
    },
}

SUBJECT_KEYS = tuple(HYPOTHESES.keys())


@dataclass
class NliScore:
    pro: float
    anti: float


class ZeroShotScorer:
    """Wraps a transformers zero-shot classification pipeline with optional batching."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int | None = None):
        self.model_name = model_name
        self.pipeline = None
        self._hypothesis_template = "{}"
        self._raw_subject_hypotheses: Dict[str, List[str]] = {
            subject: [pairs["pro"], pairs["anti"]] for subject, pairs in HYPOTHESES.items()
        }
        if pipeline is None:
            LOGGER.warning("transformers library not available; MNLI scores will default to zero.")
            return
        try:
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device if device is not None else -1,
            )
            self._hypothesis_template = getattr(self.pipeline, "hypothesis_template", "{}")
        except Exception as exc:  # pragma: no cover - depends on environment
            LOGGER.warning(
                "Failed to load zero-shot model %s (%s). MNLI scores will default to zero.",
                model_name,
                exc,
            )
            self.pipeline = None

    def score(self, text: str, subject: str) -> NliScore:
        if not text or subject not in SUBJECT_KEYS or self.pipeline is None:
            return NliScore(pro=0.0, anti=0.0)
        hypotheses = self._prepare_hypotheses(subject)
        try:
            result = self.pipeline(text, candidate_labels=hypotheses, multi_label=True)
        except Exception:
            LOGGER.warning("Zero-shot pipeline failed; returning neutral scores.")
            return NliScore(pro=0.0, anti=0.0)
        label_to_score = {label: score for label, score in zip(result["labels"], result["scores"])}
        return NliScore(
            pro=label_to_score.get(hypotheses[0], 0.0),
            anti=label_to_score.get(hypotheses[1], 0.0),
        )

    def score_batch(
        self,
        texts: List[str],
        subject: str,
        batch_size: int = 32,
        device: str = "auto",
    ) -> List[NliScore]:
        if not texts:
            return []
        if subject not in SUBJECT_KEYS:
            return [NliScore(pro=0.0, anti=0.0) for _ in texts]
        if self.pipeline is None:
            return [self.score(text, subject) for text in texts]

        hypotheses = self._prepare_hypotheses(subject)
        results: List[NliScore] = [NliScore(pro=0.0, anti=0.0) for _ in texts]
        non_empty_indices = [idx for idx, text in enumerate(texts) if text]
        if not non_empty_indices:
            return results

        for start in range(0, len(non_empty_indices), batch_size):
            batch_indices = non_empty_indices[start : start + batch_size]
            chunk_texts = [texts[idx] for idx in batch_indices]
            try:
                outputs = self.pipeline(
                    chunk_texts,
                    candidate_labels=hypotheses,
                    multi_label=True,
                    batch_size=min(batch_size, len(chunk_texts)),
                )
            except Exception:
                LOGGER.warning("Zero-shot pipeline failed during batch scoring; filling zeros.")
                outputs = [None] * len(batch_indices)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for idx_in_list, output in zip(batch_indices, outputs):
                if not output:
                    results[idx_in_list] = NliScore(pro=0.0, anti=0.0)
                    continue
                label_to_score = {label: score for label, score in zip(output["labels"], output["scores"])}
                results[idx_in_list] = NliScore(
                    pro=label_to_score.get(hypotheses[0], 0.0),
                    anti=label_to_score.get(hypotheses[1], 0.0),
                )
        return results

    def score_all_subjects_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: str = "auto",
    ) -> Dict[str, List[NliScore]]:
        return {
            subject: self.score_batch(texts, subject, batch_size=batch_size, device=device)
            for subject in SUBJECT_KEYS
        }

    def _prepare_hypotheses(self, subject: str) -> List[str]:
        raw_list = self._raw_subject_hypotheses.get(subject, [])
        template = self._hypothesis_template or "{}"
        return [template.format(label) for label in raw_list]


if __name__ == "__main__":  # pragma: no cover - debugging aid
    logging.basicConfig(level=logging.INFO)
    scorer = ZeroShotScorer()
    if scorer.pipeline is None:
        print("transformers not available; skipping self-test.")
        raise SystemExit(0)

    texts = ["I love electric cars and EVs."] * 3
    texts += [
        "Government mandates for EV adoption are necessary.",
        "Subsidies distort the market and hurt taxpayers.",
        "Mandates hurt personal freedom.",
    ]

    start_single = time.perf_counter()
    single_scores = {
        subject: [scorer.score(text, subject) for text in texts] for subject in SUBJECT_KEYS
    }
    single_time = time.perf_counter() - start_single

    start_batch = time.perf_counter()
    batch_scores = scorer.score_all_subjects_batch(texts, batch_size=4)
    batch_time = time.perf_counter() - start_batch

    for subject in SUBJECT_KEYS:
        direct_batch = scorer.score_batch(texts, subject, batch_size=4)
        for single, batch in zip(single_scores[subject], direct_batch):
            if not math.isclose(single.pro, batch.pro, abs_tol=1e-4) or not math.isclose(
                single.anti, batch.anti, abs_tol=1e-4
            ):
                raise AssertionError(f"Mismatch detected for subject {subject}")
        for combined, direct in zip(batch_scores[subject], direct_batch):
            if not math.isclose(combined.pro, direct.pro, abs_tol=1e-4) or not math.isclose(
                combined.anti, direct.anti, abs_tol=1e-4
            ):
                raise AssertionError(f"All-subject batch mismatch for subject {subject}")

    speedup = float("inf") if batch_time == 0 else single_time / batch_time
    print(
        f"Self-test passed: batched equals single within tolerance. "
        f"single={single_time:.2f}s batch={batch_time:.2f}s speedup={speedup:.2f}x"
    )
