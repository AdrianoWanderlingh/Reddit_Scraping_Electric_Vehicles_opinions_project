# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Weak-rule heuristics for EV stance scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

SIMPLE_RULES = {
    "product": {
        "pro": [
            "love my ev",
            "buy an ev",
            "better for the environment",
        ],
        "anti": [
            "range anxiety",
            "battery fire",
            "too expensive",
        ],
    },
    "mandate": {
        "pro": [
            "support the mandate",
            "ban gas cars",
            "need the requirement",
        ],
        "anti": [
            "government mandate",
            "forced to buy",
            "against the ban",
        ],
    },
    "policy": {
        "pro": [
            "tax credit",
            "subsidies help",
            "policy works",
        ],
        "anti": [
            "waste of taxpayers",
            "policy failure",
            "subsidies distort",
        ],
    },
}

FULL_RULE_KEYS = {
    "pro": "pro_cues",
    "anti": "anti_cues",
}

SUBJECTS = ("product", "mandate", "policy")


@dataclass
class WeakRuleScore:
    pro: float
    anti: float
    pro_hits: int
    anti_hits: int


class WeakRuleScorer:
    """Phrase-based heuristics for stance scoring."""

    def __init__(
        self,
        keyword_config: Mapping[str, Mapping[str, Sequence[str]]] | None,
        mode: str = "simple",
    ) -> None:
        self.mode = mode
        self.keyword_config = keyword_config or {}
        if mode == "full":
            self.pro_cues = self._load_full_rules("pro")
            self.anti_cues = self._load_full_rules("anti")
        else:
            self.pro_cues = {subject: SIMPLE_RULES[subject]["pro"] for subject in SUBJECTS}
            self.anti_cues = {subject: SIMPLE_RULES[subject]["anti"] for subject in SUBJECTS}

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

    def _load_full_rules(self, stance: str) -> Dict[str, Sequence[str]]:
        key = FULL_RULE_KEYS[stance]
        config_block = self.keyword_config.get(key, {})
        result: Dict[str, Sequence[str]] = {}
        for subject in SUBJECTS:
            values = config_block.get(subject, ()) if isinstance(config_block, Mapping) else ()
            result[subject] = tuple(values) if values else ()
        return result

    @staticmethod
    def _count_hits(text: str, phrases: Sequence[str]) -> int:
        if not text or not phrases:
            return 0
        total = 0
        for phrase in phrases:
            token = phrase.lower().strip()
            if not token:
                continue
            total += text.count(token)
        return total

    @staticmethod
    def _squash(count: int) -> float:
        return 1.0 - math.exp(-count) if count > 0 else 0.0
