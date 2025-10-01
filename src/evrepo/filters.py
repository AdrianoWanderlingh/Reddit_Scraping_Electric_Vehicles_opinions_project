from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class CandidateFilter:
    positive_patterns: List[re.Pattern]
    context_patterns: List[re.Pattern]
    negative_patterns: List[re.Pattern]
    require_context: bool = False

    @classmethod
    def from_config(cls, keywords: dict | None, neg_filters: dict | Sequence[str] | None) -> "CandidateFilter":
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
    if term.startswith("regex:"):
        pattern = term[len("regex:"):].strip()
        return re.compile(pattern, re.IGNORECASE)
    return re.compile(rf"(?i)\b{re.escape(term)}\b")


def _looks_english(text: str) -> bool:
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if alpha == 0:
        return False
    return non_ascii <= alpha * 0.5
