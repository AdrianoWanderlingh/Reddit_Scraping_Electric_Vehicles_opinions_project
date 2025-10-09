#!/usr/bin/env python
"""Deprecated: use scripts/quick_test.py instead."""
from __future__ import annotations

import warnings

from quick_test import main as quick_test_main


if __name__ == "__main__":
    warnings.warn("scripts/run_pipeline_quick.py is deprecated; use scripts/quick_test.py", DeprecationWarning, stacklevel=2)
    quick_test_main()
