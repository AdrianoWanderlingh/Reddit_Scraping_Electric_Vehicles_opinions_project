# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Subject scoring utilities for EV stance detection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

SUBJECTS: Sequence[str] = ("product", "mandate", "policy")


@dataclass
class SubjectScores:
    """Container for raw hit counts and normalized scores per subject."""

    raw: Dict[str, float]
    normalized: Dict[str, float]

    def primary(self, priority: Sequence[str]) -> str:
        """Return the subject with the highest normalized score.

        The *priority* sequence breaks ties deterministically.
        """

        best_subject = priority[0] if priority else SUBJECTS[0]
        best_score = self.normalized.get(best_subject, 0.0)
        for subject, score in self.normalized.items():
            if score > best_score:
                best_subject, best_score = subject, score
            elif score == best_score and subject in priority:
                if priority.index(subject) < priority.index(best_subject):
                    best_subject = subject
        return best_subject


class SubjectScorer:
    """Scores text for product, mandate, and policy relevance based on regex keywords."""

    def __init__(self, keyword_config: Mapping[str, Iterable[str]] | None):
        self.keyword_config = keyword_config or {}
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
        compiled: Dict[str, Dict[str, List[re.Pattern]]] = {
            "product": {"core": [], "context": []},
            "mandate": {"core": []},
            "policy": {"core": []},
        }
        for key in ("product_core", "product_context"):
            for pattern in self.keyword_config.get(key, []) or []:
                compiled["product"]["core" if key == "product_core" else "context"].append(
                    re.compile(pattern, re.IGNORECASE)
                )
        for pattern in self.keyword_config.get("mandate", []) or []:
            compiled["mandate"]["core"].append(re.compile(pattern, re.IGNORECASE))
        for pattern in self.keyword_config.get("policy_non_mandate", []) or []:
            compiled["policy"]["core"].append(re.compile(pattern, re.IGNORECASE))
        return compiled

    def score(self, text: str) -> SubjectScores:
        """Return raw and normalized scores for each subject."""

        lowered = text.lower() if text else ""
        raw: Dict[str, float] = {subject: 0.0 for subject in SUBJECTS}
        normalized: Dict[str, float] = {subject: 0.0 for subject in SUBJECTS}

        if not lowered:
            return SubjectScores(raw=raw, normalized=normalized)

        # Product subject: emphasize core phrases, lightly reward context terms.
        product_core_hits = self._count_matches(lowered, self.patterns["product"]["core"])
        product_context_hits = self._count_matches(lowered, self.patterns["product"].get("context", []))
        raw["product"] = product_core_hits + 0.5 * product_context_hits

        # Mandate and policy rely solely on their core lists for now.
        raw["mandate"] = self._count_matches(lowered, self.patterns["mandate"]["core"])
        raw["policy"] = self._count_matches(lowered, self.patterns["policy"]["core"])

        for subject, value in raw.items():
            normalized[subject] = 1.0 - math.exp(-value) if value > 0 else 0.0

        return SubjectScores(raw=raw, normalized=normalized)

    @staticmethod
    def _count_matches(text: str, patterns: Sequence[re.Pattern]) -> float:
        count = 0.0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count


def reweight_scores(scores: Dict[str, float], subject: str, bonus: float) -> Dict[str, float]:
    """Utility used by tests: manually adjust a subject score by *bonus*."""

    updated = dict(scores)
    updated[subject] = updated.get(subject, 0.0) + bonus
    return updated
