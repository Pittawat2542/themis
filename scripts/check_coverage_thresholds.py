#!/usr/bin/env python3
"""Enforce module-level coverage thresholds from a coverage JSON report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "themis/api.py": 75.0,
    "themis/experiment/orchestrator.py": 75.0,
    "themis/experiment/storage.py": 80.0,
    "themis/comparison/engine.py": 80.0,
    "themis/evaluation/statistics/comparison_tests.py": 80.0,
    "themis/evaluation/metrics/code/execution.py": 50.0,
}


def main(argv: list[str]) -> int:
    coverage_path = Path(argv[1]) if len(argv) > 1 else Path("coverage.json")
    if not coverage_path.exists():
        print(f"[coverage-gate] coverage file not found: {coverage_path}")
        return 2

    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})

    failures: list[str] = []
    for module, threshold in THRESHOLDS.items():
        summary = files.get(module, {}).get("summary")
        if summary is None:
            failures.append(f"{module}: missing from coverage report")
            continue
        percent = float(summary.get("percent_covered", 0.0))
        if percent < threshold:
            failures.append(
                f"{module}: {percent:.2f}% < required {threshold:.2f}%"
            )
        else:
            print(
                f"[coverage-gate] {module}: {percent:.2f}% "
                f"(threshold {threshold:.2f}%)"
            )

    if failures:
        print("[coverage-gate] Module coverage check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[coverage-gate] All module thresholds satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
