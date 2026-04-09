"""Inspection CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.cli.helpers import dump_json, initialize_store, load_experiment
from themis.core.inspection import (
    get_evaluation_execution,
    get_execution_state,
    get_run_snapshot,
)

inspect_app = App(
    name="inspect", help="Inspect persisted snapshots and execution state."
)


@inspect_app.command
def snapshot(*, config: str) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    print(
        dump_json(
            get_run_snapshot(store, experiment.compile().run_id).model_dump(mode="json")
        )
    )
    return 0


@inspect_app.command
def state(*, config: str) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    print(
        dump_json(
            get_execution_state(store, experiment.compile().run_id).model_dump(
                mode="json"
            )
        )
    )
    return 0


@inspect_app.command
def evaluation(
    *,
    config: str,
    case_id: str,
    metric_id: str,
    dataset_id: str | None = None,
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    execution = get_evaluation_execution(
        store,
        experiment.compile().run_id,
        case_id,
        metric_id,
        dataset_id=dataset_id,
    )
    if execution is None:
        raise SystemExit(
            f"No evaluation execution found for case_id={case_id} metric_id={metric_id}"
        )
    print(dump_json(execution.model_dump(mode="json")))
    return 0
