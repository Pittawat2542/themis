#!/usr/bin/env python3
"""Enforce module-level coverage thresholds from a coverage JSON report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "themis/cli/quickcheck.py": 80.0,
    "themis/extractors/builtin.py": 80.0,
    "themis/orchestration/orchestrator.py": 90.0,
    "themis/registry/plugin_registry.py": 75.0,
    "themis/stats/stats_engine.py": 85.0,
    "themis/storage/projection_repo.py": 70.0,
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
            failures.append(f"{module}: {percent:.2f}% < required {threshold:.2f}%")
        else:
            print(
                f"[coverage-gate] {module}: {percent:.2f}% (threshold {threshold:.2f}%)"
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
