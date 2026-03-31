"""Run-oriented CLI commands."""

from __future__ import annotations

from typing import Literal

from themis.cli.helpers import dump_json, initialize_store, load_experiment
from themis.core.planner import Planner


def run(*, config: str) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    result = experiment.run(store=store)
    report_store = store if experiment.storage.store == "memory" else initialize_store(experiment)
    benchmark = report_store.get_projection(result.run_id, "benchmark_result")
    metric_means = {}
    if isinstance(benchmark, dict):
        metric_means = dict(benchmark.get("metric_means", {}))
    print(
        dump_json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "metric_means": metric_means,
            }
        )
    )
    return 0


def resume(*, config: str) -> int:
    experiment = load_experiment(config)
    snapshot = experiment.compile()
    store = initialize_store(experiment)
    stored = store.resume(snapshot.run_id)
    if stored is None:
        raise SystemExit(f"Unknown run_id: {snapshot.run_id}")
    print(
        dump_json(
            {
                "run_id": snapshot.run_id,
                "status": stored.execution_state.status.value,
                "total_cases": sum(len(dataset.cases) for dataset in stored.snapshot.datasets),
                "completed_cases": len(stored.execution_state.case_states),
            }
        )
    )
    return 0


def estimate(*, config: str) -> int:
    experiment = load_experiment(config)
    estimate_result = Planner().estimate(experiment.compile())
    print(dump_json(estimate_result.model_dump(mode="json")))
    return 0


def quickcheck(*, config: str) -> int:
    experiment = load_experiment(config)
    snapshot = experiment.compile()
    store = initialize_store(experiment)
    from themis.core.quickcheck import quickcheck as quickcheck_run

    print(dump_json(quickcheck_run(store, snapshot.run_id)))
    return 0


def replay(
    *,
    config: str,
    stage: Literal["reduce", "parse", "score", "judge"],
    metric_id: list[str] | None = None,
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    result = experiment.replay(stage=stage, metric_ids=metric_id, store=store)
    benchmark = store.get_projection(result.run_id, "benchmark_result")
    metric_means = {}
    if isinstance(benchmark, dict):
        metric_means = dict(benchmark.get("metric_means", {}))
    print(
        dump_json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "metric_means": metric_means,
            }
        )
    )
    return 0
