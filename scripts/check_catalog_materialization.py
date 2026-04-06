#!/usr/bin/env python3
"""Exhaustively materialize shipped catalog benchmarks.

Requires `themis-eval[datasets]`, network access, and any needed Hugging Face
authentication for private or gated datasets.
"""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys
from typing import NamedTuple, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from themis.catalog import load
from themis.catalog.benchmarks import (
    BenchmarkDefinition,
    _materialization_benchmark_ids,
)


class MaterializationCheckResult(NamedTuple):
    benchmark_id: str
    status: str
    case_count: int | None = None
    message: str | None = None


def check_catalog_materialization(
    benchmark_ids: Sequence[str] | None = None,
) -> list[MaterializationCheckResult]:
    targets = list(
        _materialization_benchmark_ids() if benchmark_ids is None else benchmark_ids
    )
    results: list[MaterializationCheckResult] = []

    for benchmark_id in targets:
        try:
            benchmark = cast(BenchmarkDefinition, load(benchmark_id))
            dataset = benchmark.materialize_dataset()
        except Exception as exc:
            results.append(
                MaterializationCheckResult(
                    benchmark_id=benchmark_id,
                    status="failed",
                    message=str(exc),
                )
            )
            continue

        case_count = len(dataset.cases)
        if case_count == 0:
            results.append(
                MaterializationCheckResult(
                    benchmark_id=benchmark_id,
                    status="failed",
                    case_count=0,
                    message="Benchmark materialized an empty dataset.",
                )
            )
            continue

        results.append(
            MaterializationCheckResult(
                benchmark_id=benchmark_id,
                status="passed",
                case_count=case_count,
            )
        )

    return results


def main() -> int:
    results = check_catalog_materialization()
    passed = sum(result.status == "passed" for result in results)
    failed_results = [result for result in results if result.status != "passed"]

    for result in results:
        if result.status == "passed":
            assert result.case_count is not None
            print(f"PASS {result.benchmark_id} ({result.case_count} cases)")
        else:
            print(f"FAIL {result.benchmark_id} ({result.message or 'unknown error'})")

    print(
        f"Checked {len(results)} benchmarks: {passed} passed, "
        f"{len(failed_results)} failed."
    )
    if failed_results:
        print("Failures:")
        for result in failed_results:
            print(f"{result.benchmark_id}: {result.message or 'unknown error'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
