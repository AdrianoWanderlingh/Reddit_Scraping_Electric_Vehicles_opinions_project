# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Keyword-based EV candidate filtering utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

__all__ = ["CandidateFilter", "reweight_scores"]


def _flatten_terms(value) -> List[str]:
    """Recursively flatten strings / iterables into a simple string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    result: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            result.extend(_flatten_terms(item))
        return result
    if isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_terms(item))
        return result
    return result


def _compile(term: str) -> re.Pattern:
    """Compile *term* into a case-insensitive regex, honouring explicit regex markers."""
    if term.startswith("regex:"):
        pattern = term[len("regex:") :].strip()
        return re.compile(pattern, re.IGNORECASE)
    return re.compile(term, re.IGNORECASE)





def _looks_english(text: str) -> bool:
    """Basic heuristic to reject obviously non-English strings."""
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if alpha == 0:
        return False
    return non_ascii <= alpha * 0.5


@dataclass
class CandidateFilter:
    """Container for compiled regex patterns that flag EV-related text."""

    positive_patterns: List[re.Pattern]
    context_patterns: List[re.Pattern]
    negative_patterns: List[re.Pattern]
    require_context: bool = False

    @classmethod
    def from_config(cls, keywords: dict | None, neg_filters: dict | Sequence[str] | None) -> "CandidateFilter":
        """Build a filter from config dictionaries of positive/context/negative terms."""

        keywords = keywords or {}
        product_core_terms = _flatten_terms(keywords.get("product_core"))
        product_context_terms = _flatten_terms(keywords.get("product_context"))

        negative_terms: List[str] = []
        if isinstance(neg_filters, dict):
            negative_terms = _flatten_terms(neg_filters.get("drop_regex"))
        else:
            negative_terms = _flatten_terms(neg_filters)

        positive_patterns = [_compile(term) for term in product_core_terms if term]
        context_patterns = [_compile(term) for term in product_context_terms if term]
        negative_patterns = [_compile(term) for term in negative_terms if term]

        require_context = bool(context_patterns) and bool(positive_patterns)

        return cls(
            positive_patterns=positive_patterns or [_compile("electric vehicle")],
            context_patterns=context_patterns,
            negative_patterns=negative_patterns,
            require_context=require_context,
        )

    def is_candidate(self, text: str) -> bool:
        """Return True when the text should be treated as EV-related."""
        if not text:
            return False
        text = text.lower()
        if any(pattern.search(text) for pattern in self.negative_patterns):
            return False
        has_positive = any(pattern.search(text) for pattern in self.positive_patterns)
        if not has_positive:
            return False
        if self.require_context and not any(pattern.search(text) for pattern in self.context_patterns):
            return False
        if not _looks_english(text):
            return False
        return True


def reweight_scores(scores: Dict[str, float], subject: str, bonus: float) -> Dict[str, float]:
    """Utility used by tests: manually adjust a subject score by *bonus*."""

    updated = dict(scores)
    updated[subject] = updated.get(subject, 0.0) + bonus
    return updated

