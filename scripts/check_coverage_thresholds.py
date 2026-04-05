from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


MODULE_THRESHOLDS = {
    "themis/adapters": 75.0,
    "themis/catalog": 85.0,
    "themis/core": 85.0,
}


def _number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _coverage_percent(summary: dict[str, object]) -> float:
    covered = _number(summary.get("covered_lines", 0))
    total = _number(summary.get("num_statements", 0))
    if total == 0:
        return 100.0
    return covered / total * 100.0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_coverage_thresholds.py <coverage.json>", file=sys.stderr)
        return 2

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    files = payload.get("files", {})
    module_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    failures: list[str] = []

    for filename, entry in sorted(files.items()):
        if not isinstance(filename, str) or not filename.startswith("themis/"):
            continue
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary", {})
        if not isinstance(summary, dict):
            continue

        parts = filename.split("/")
        module = "/".join(parts[:2]) if len(parts) > 1 else filename
        if module not in MODULE_THRESHOLDS:
            continue

        module_stats[module][0] += _number(summary.get("covered_lines", 0))
        module_stats[module][1] += _number(summary.get("num_statements", 0))

    for module, threshold in MODULE_THRESHOLDS.items():
        covered, total = module_stats[module]
        percent = 100.0 if total == 0 else covered / total * 100.0
        if percent < threshold:
            failures.append(f"{module}: {percent:.1f}% < {threshold:.1f}%")

    if failures:
        print("coverage threshold failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("module-level coverage thresholds satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
