"""Comparison CLI commands."""

from __future__ import annotations

from themis.cli.helpers import (
    build_run_query,
    dump_json,
    initialize_store,
    load_benchmark_result,
    load_experiment,
    resolve_persisted_run_id,
)
from themis.core.stats import StatsEngine


def compare(
    *,
    baseline_config: str,
    candidate_config: str,
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    baseline_baseline_label: str | None = None,
    candidate_baseline_label: str | None = None,
) -> int:
    baseline_experiment = load_experiment(baseline_config)
    candidate_experiment = load_experiment(candidate_config)
    baseline_store = initialize_store(baseline_experiment)
    candidate_store = initialize_store(candidate_experiment)
    resolved_baseline_run_id = (
        resolve_persisted_run_id(
            baseline_store,
            run_id=baseline_run_id,
            baseline_label=baseline_baseline_label,
            query=build_run_query(),
        )
        if baseline_run_id is not None or baseline_baseline_label is not None
        else baseline_experiment.compile().run_id
    )
    resolved_candidate_run_id = (
        resolve_persisted_run_id(
            candidate_store,
            run_id=candidate_run_id,
            baseline_label=candidate_baseline_label,
            query=build_run_query(),
        )
        if candidate_run_id is not None or candidate_baseline_label is not None
        else candidate_experiment.compile().run_id
    )
    comparison = StatsEngine().paired_compare(
        load_benchmark_result(baseline_store, resolved_baseline_run_id),
        load_benchmark_result(candidate_store, resolved_candidate_run_id),
    )
    print(dump_json(comparison))
    return 0
