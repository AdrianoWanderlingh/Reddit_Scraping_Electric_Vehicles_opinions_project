#!/usr/bin/env python
"""Deprecated wrapper for score_sentiment.py."""
from __future__ import annotations

import warnings

from score_sentiment import main as score_sentiment_main


def main() -> None:
    warnings.warn(
        "scripts/run_sentiment_sample.py is deprecated; use scripts/score_sentiment.py",
        DeprecationWarning,
        stacklevel=2,
    )
    score_sentiment_main()


if __name__ == "__main__":
    main()
