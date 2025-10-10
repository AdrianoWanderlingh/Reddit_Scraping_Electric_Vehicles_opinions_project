# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Combine subject, weak-rule, and MNLI scores into final stance labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .subjects import SUBJECTS, SubjectScores
from .weak_rules import WeakRuleScore
from .nli_zeroshot import NliScore

STANCE_MAP = {
    ("product", "pro"): "Pro-EV (product)",
    ("product", "anti"): "Against-EV (product)",
    ("product", "neutral"): "Neutral-EV (product)",
    ("mandate", "pro"): "Pro-Mandate",
    ("mandate", "anti"): "Against-Mandate",
    ("mandate", "neutral"): "Neutral-Mandate",
    ("policy", "pro"): "Pro-Policy",
    ("policy", "anti"): "Against-Policy",
    ("policy", "neutral"): "Neutral-Policy",
}


@dataclass
class LabelDecision:
    subject: str
    stance: str
    category: str
    confidence: float
    fused_pro: float
    fused_anti: float


class LabelFuser:
    """Apply weighted fusion between weak rules and MNLI scores."""

    def __init__(
        self,
        fusion_config: Mapping[str, float] | None,
        subject_priority: Sequence[str] | None,
    ):
        fusion_config = fusion_config or {}
        self.weak_weight = fusion_config.get("weak_weight", 0.4)
        self.nli_weight = fusion_config.get("nli_weight", 0.6)
        total = self.weak_weight + self.nli_weight
        if total == 0:
            self.weak_weight = self.nli_weight = 0.5
            total = 1
        self.weak_weight /= total
        self.nli_weight /= total
        self.neutral_threshold = fusion_config.get("neutral_threshold", 0.55)
        self.subject_priority = list(subject_priority or SUBJECTS)

    def fuse(
        self,
        subject_scores: SubjectScores,
        weak_scores: Dict[str, WeakRuleScore],
        nli_scores: Dict[str, NliScore],
    ) -> LabelDecision:
        subject = subject_scores.primary(self.subject_priority)
        weak = weak_scores.get(subject) or WeakRuleScore(0.0, 0.0, 0, 0)
        nli = nli_scores.get(subject) or NliScore(0.0, 0.0)

        fused_pro = self._combine(weak.pro, nli.pro)
        fused_anti = self._combine(weak.anti, nli.anti)

        if max(fused_pro, fused_anti) < self.neutral_threshold:
            stance = "neutral"
        else:
            stance = "pro" if fused_pro >= fused_anti else "anti"

        category = STANCE_MAP[(subject, stance)]
        confidence = abs(fused_pro - fused_anti)
        return LabelDecision(
            subject=subject,
            stance=stance,
            category=category,
            confidence=confidence,
            fused_pro=fused_pro,
            fused_anti=fused_anti,
        )

    def _combine(self, weak: float, nli: float) -> float:
        return self.weak_weight * weak + self.nli_weight * nli

    def decide_nli_only(
        self,
        subject_scores: SubjectScores,
        nli_scores: Dict[str, NliScore],
    ) -> LabelDecision:
        """Fast-path decision using NLI only (no weak-rule fusion).

        - Chooses primary subject using subject_scores and configured tie-break priority.
        - Applies neutral threshold to the subject's (pro, anti) entailment probabilities.
        - Returns a LabelDecision with fused_* equal to the raw NLI probabilities.
        """
        subject = subject_scores.primary(self.subject_priority)
        nli = nli_scores.get(subject) or NliScore(0.0, 0.0)
        pro, anti = nli.pro, nli.anti
        if max(pro, anti) < self.neutral_threshold:
            stance = "neutral"
        else:
            stance = "pro" if pro >= anti else "anti"
        category = STANCE_MAP[(subject, stance)]
        confidence = abs(pro - anti)
        return LabelDecision(
            subject=subject,
            stance=stance,
            category=category,
            confidence=confidence,
            fused_pro=pro,
            fused_anti=anti,
        )
