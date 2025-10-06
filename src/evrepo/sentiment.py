# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Sentiment scoring utilities (VADER + transformer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:  # transformers is optional at import time
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - external dependency missing
    pipeline = None  # type: ignore

LOGGER = logging.getLogger("evrepo.sentiment")


@dataclass
class SentimentScore:
    vader_compound: float
    transformer_label: str
    transformer_score: float


class SentimentAnalyzer:
    """Compute VADER compound and transformer sentiment for free text."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[int] = None,
    ) -> None:
        self.vader = SentimentIntensityAnalyzer()
        self.model_name = model_name
        self.transformer = None
        if pipeline is None:
            LOGGER.warning(
                "transformers library not available; transformer sentiment scores will default to neutral."
            )
            return
        try:
            self.transformer = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device if device is not None else -1,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Failed to load sentiment model %s (%s). Transformer scores will be neutral.",
                model_name,
                exc,
            )
            self.transformer = None

    def score(self, text: str | None) -> SentimentScore:
        if not text:
            return SentimentScore(vader_compound=0.0, transformer_label="neutral", transformer_score=0.0)

        vader_compound = self.vader.polarity_scores(text)["compound"]

        if self.transformer is None:
            return SentimentScore(vader_compound=vader_compound, transformer_label="neutral", transformer_score=0.0)

        try:
            result = self.transformer(text, truncation=True)[0]
            label = result.get("label", "neutral").lower()
            score = float(result.get("score", 0.0))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Transformer sentiment pipeline failed (%s); returning neutral.", exc)
            label, score = "neutral", 0.0

        return SentimentScore(
            vader_compound=vader_compound,
            transformer_label=label,
            transformer_score=score,
        )
