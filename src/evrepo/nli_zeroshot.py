# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Zero-shot MNLI scoring using HuggingFace transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

try:  # Delay heavy imports until needed.
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
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


@dataclass
class NliScore:
    pro: float
    anti: float


class ZeroShotScorer:
    """Wraps a transformers zero-shot classification pipeline with safe fallbacks."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int | None = None):
        self.model_name = model_name
        self.pipeline = None
        if pipeline is None:
            LOGGER.warning("transformers library not available; MNLI scores will default to zero.")
            return
        try:
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device if device is not None else -1,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            LOGGER.warning(
                "Failed to load zero-shot model %s (%s). MNLI scores will be zero.",
                model_name,
                exc,
            )
            self.pipeline = None

    def score(self, text: str, subject: str) -> NliScore:
        if not text:
            return NliScore(pro=0.0, anti=0.0)
        hypotheses = HYPOTHESES.get(subject)
        if not hypotheses:
            return NliScore(pro=0.0, anti=0.0)
        if self.pipeline is None:
            return NliScore(pro=0.0, anti=0.0)

        candidate_labels = [hypotheses["pro"], hypotheses["anti"]]
        try:
            result = self.pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=True,
            )
        except Exception as exc:  # pragma: no cover - runtime issues
            LOGGER.warning("Zero-shot pipeline failed (%s); returning neutral scores.", exc)
            return NliScore(pro=0.0, anti=0.0)

        scores = {label: score for label, score in zip(result["labels"], result["scores"])}
        return NliScore(
            pro=scores.get(hypotheses["pro"], 0.0),
            anti=scores.get(hypotheses["anti"], 0.0),
        )
