# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Keyword-based EV candidate filtering utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


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
        positive = _collect_terms(keywords)
        context = _collect_terms(keywords, keys=("context",))
        negative = _collect_terms(neg_filters)

        positive_patterns = [_compile(term) for term in positive if term]
        context_patterns = [_compile(term) for term in context if term]
        negative_patterns = [_compile(term) for term in negative if term]

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


def _collect_terms(config: dict | Sequence[str] | None, keys: Sequence[str] | None = None) -> List[str]:
    """Recursively flatten a config object into a list of string terms."""
    terms: List[str] = []
    if config is None:
        return terms
    if isinstance(config, str):
        return [config]
    if isinstance(config, (list, tuple, set)):
        for item in config:
            terms.extend(_collect_terms(item, keys))
        return terms
    if isinstance(config, dict):
        for key, value in config.items():
            key_lc = str(key).lower()
            if keys is None or any(k in key_lc for k in keys):
                terms.extend(_collect_terms(value, None))
            else:
                terms.extend(_collect_terms(value, keys))
        return terms
    return terms


def _compile(term: str) -> re.Pattern:
    """Compile *term* into a case-insensitive regex, honoring literal 'regex:' prefixes."""
    if term.startswith("regex:"):
        pattern = term[len("regex:"):].strip()
        return re.compile(pattern, re.IGNORECASE)
    return re.compile(rf"(?i)\b{re.escape(term)}\b")


def _looks_english(text: str) -> bool:
    """Filter out obvious non-English strings using a basic ASCII heuristic."""
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if alpha == 0:
        return False
    return non_ascii <= alpha * 0.5
