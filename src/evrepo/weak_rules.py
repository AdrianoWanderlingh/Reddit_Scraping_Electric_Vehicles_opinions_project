# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Weak rule scoring for EV stance detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

SUBJECTS = ("product", "mandate", "policy")


@dataclass
class WeakRuleScore:
    """Per-subject weak rule scores and hit counts."""

    pro: float
    anti: float
    pro_hits: int
    anti_hits: int


class WeakRuleScorer:
    """Scores pro/anti cues using configurable lexicons."""

    def __init__(self, keyword_config: Mapping[str, Mapping[str, Sequence[str]]] | None):
        self.keyword_config = keyword_config or {}
        cues = self.keyword_config.get("pro_cues", {})
        anti = self.keyword_config.get("anti_cues", {})
        self.pro_cues = {subject: tuple(v or []) for subject, v in cues.items()}
        self.anti_cues = {subject: tuple(v or []) for subject, v in anti.items()}

    def score(self, text: str) -> Dict[str, WeakRuleScore]:
        lowered = text.lower() if text else ""
        scores: Dict[str, WeakRuleScore] = {}
        for subject in SUBJECTS:
            pro_hits = self._count_hits(lowered, self.pro_cues.get(subject, ()))
            anti_hits = self._count_hits(lowered, self.anti_cues.get(subject, ()))
            scores[subject] = WeakRuleScore(
                pro=self._squash(pro_hits),
                anti=self._squash(anti_hits),
                pro_hits=pro_hits,
                anti_hits=anti_hits,
            )
        return scores

    @staticmethod
    def _count_hits(text: str, phrases: Sequence[str]) -> int:
        if not text or not phrases:
            return 0
        total = 0
        for phrase in phrases:
            if not phrase:
                continue
            total += text.count(phrase.lower())
        return total

    @staticmethod
    def _squash(count: int) -> float:
        """Map a raw hit count onto (0, 1) with diminishing returns."""
        return 1.0 - math.exp(-count) if count > 0 else 0.0
