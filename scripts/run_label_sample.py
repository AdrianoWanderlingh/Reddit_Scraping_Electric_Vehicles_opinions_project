#!/usr/bin/env python
"""Deprecated wrapper for label_stance.py."""
from __future__ import annotations

import warnings

from label_stance import main as label_stance_main


def main() -> None:
    warnings.warn(
        "scripts/run_label_sample.py is deprecated; use scripts/label_stance.py",
        DeprecationWarning,
        stacklevel=2,
    )
    label_stance_main()


if __name__ == "__main__":
    main()
