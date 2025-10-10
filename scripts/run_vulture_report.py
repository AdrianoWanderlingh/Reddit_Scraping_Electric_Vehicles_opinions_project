#!/usr/bin/env python
"""Run vulture dead-code analysis and store report."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPORT_PATH = Path("vulture_report.txt")


def main() -> None:
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        subprocess.run(["vulture", "src", "scripts"], check=True, stdout=handle)
    print(f"Vulture report written to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
