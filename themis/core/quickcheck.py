"""Quick read-side summaries for stored runs."""

from __future__ import annotations

from themis.core.base import JSONValue
from themis.core.reporter import Reporter
from themis.core.store import RunStore


def quickcheck(store: RunStore, run_id: str) -> dict[str, JSONValue]:
    reporter = Reporter(store)
    run_result = store.get_projection(run_id, "run_result")
    benchmark_result = store.get_projection(run_id, "benchmark_result")
    if not isinstance(run_result, dict) or not isinstance(benchmark_result, dict):
        raise ValueError(f"Run projections unavailable for run_id={run_id}")
    return {
        "run_id": run_id,
        "status": run_result["status"],
        "total_cases": run_result["progress"]["total_cases"],
        "completed_cases": run_result["progress"]["completed_cases"],
        "failed_cases": run_result["progress"]["failed_cases"],
        "metric_means": benchmark_result["metric_means"],
        "score_rows": reporter.export_score_table(run_id),
    }
