"""Comparison CLI commands."""

from __future__ import annotations

from themis.cli.helpers import (
    dump_json,
    initialize_store,
    load_benchmark_result,
    load_experiment,
)
from themis.core.stats import StatsEngine


def compare(*, baseline_config: str, candidate_config: str) -> int:
    baseline_experiment = load_experiment(baseline_config)
    candidate_experiment = load_experiment(candidate_config)
    baseline_store = initialize_store(baseline_experiment)
    candidate_store = initialize_store(candidate_experiment)
    comparison = StatsEngine().paired_compare(
        load_benchmark_result(baseline_store, baseline_experiment.compile().run_id),
        load_benchmark_result(candidate_store, candidate_experiment.compile().run_id),
    )
    print(dump_json(comparison))
    return 0
