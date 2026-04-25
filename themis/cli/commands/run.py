"""Run-oriented CLI commands."""

from __future__ import annotations

from typing import Literal

from themis.cli.helpers import dump_json, initialize_store, load_experiment
from themis.core.planner import Planner


def run(
    *,
    config: str,
    until_stage: Literal["generate", "reduce", "parse", "score", "judge"] = "judge",
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    result = experiment.run(store=store, until_stage=until_stage)
    report_store = (
        store if experiment.storage.target == "memory" else initialize_store(experiment)
    )
    benchmark = report_store.get_projection(result.run_id, "benchmark_result")
    metric_means = {}
    if isinstance(benchmark, dict):
        metric_means_payload = benchmark.get("metric_means", {})
        if isinstance(metric_means_payload, dict):
            metric_means = dict(metric_means_payload)
    print(
        dump_json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "completed_through_stage": result.completed_through_stage,
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
                "completed_through_stage": stored.execution_state.completed_through_stage,
                "total_cases": sum(
                    len(dataset.cases) for dataset in stored.snapshot.datasets
                ),
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
        metric_means_payload = benchmark.get("metric_means", {})
        if isinstance(metric_means_payload, dict):
            metric_means = dict(metric_means_payload)
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


def rerun(
    *,
    config: str,
    stage: Literal["generate", "reduce", "parse", "score", "judge"],
    failed_only: bool = False,
    case_id: list[str] | None = None,
    case_key: list[str] | None = None,
    metadata: list[str] | None = None,
    metric_id: list[str] | None = None,
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    metadata_filter = _metadata_filter_from_cli(metadata or [])
    result = experiment.rerun(
        stage=stage,
        failed_only=failed_only,
        case_ids=case_id or [],
        case_keys=case_key or [],
        metadata=metadata_filter,
        metric_ids=metric_id or [],
        store=store,
    )
    benchmark = store.get_projection(result.run_id, "benchmark_result")
    metric_means = {}
    if isinstance(benchmark, dict):
        metric_means_payload = benchmark.get("metric_means", {})
        if isinstance(metric_means_payload, dict):
            metric_means = dict(metric_means_payload)
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


def _metadata_filter_from_cli(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Metadata filters must use key=value syntax, got {value!r}"
            )
        key, filter_value = value.split("=", 1)
        filters[key] = filter_value
    return filters
