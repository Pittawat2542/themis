"""Quick read-side summaries for stored runs."""

from __future__ import annotations

from typing import cast

from themis.core.base import JSONValue
from themis.core.planner import Planner
from themis.core.reporter import Reporter
from themis.core.store import RunStore


def quickcheck(store: RunStore, run_id: str) -> dict[str, JSONValue]:
    reporter = Reporter(store)
    run_result = store.get_projection(run_id, "run_result")
    benchmark_result = store.get_projection(run_id, "benchmark_result")
    if not isinstance(run_result, dict) or not isinstance(benchmark_result, dict):
        raise ValueError(f"Run projections unavailable for run_id={run_id}")
    progress = _require_mapping(run_result.get("progress"), name="run_result.progress")
    score_rows = cast(JSONValue, reporter.score_rows(run_id))
    stored = store.resume(run_id)
    resource_plan = (
        Planner()
        .resource_plan(stored.snapshot, stored.snapshot.provenance.runtime)
        .model_dump(mode="json")
        if stored is not None
        else None
    )
    return {
        "run_id": run_id,
        "status": run_result["status"],
        "total_cases": progress["total_cases"],
        "completed_cases": progress["completed_cases"],
        "failed_cases": progress["failed_cases"],
        "metric_means": benchmark_result["metric_means"],
        "resource_plan": cast(JSONValue, resource_plan),
        "score_rows": score_rows,
    }


def _require_mapping(value: JSONValue | None, *, name: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object projection value for {name}")
    return value
