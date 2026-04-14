"""Inspection CLI commands."""

from __future__ import annotations

from cyclopts import App

from themis.cli.helpers import (
    build_run_query,
    dump_json,
    initialize_store,
    load_experiment,
    resolve_persisted_run_id,
)
from themis.core.inspection import (
    get_case_audit,
    get_evaluation_execution,
    get_execution_state,
    get_run_record,
    get_run_snapshot,
    get_telemetry_summary,
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
def runs(
    *,
    config: str,
    run_id: str | None = None,
    dataset_source_id: str | None = None,
    dataset_fingerprint: str | None = None,
    metric_id: str | None = None,
    tag: list[str] | None = None,
    baseline_label: str | None = None,
    lineage_parent_run_id: str | None = None,
    status: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    query = build_run_query(
        run_id=run_id,
        dataset_source_id=dataset_source_id,
        dataset_fingerprint=dataset_fingerprint,
        metric_id=metric_id,
        tags=tag,
        baseline_label=baseline_label,
        lineage_parent_run_id=lineage_parent_run_id,
        status=status,
        created_after=created_after,
        created_before=created_before,
        updated_after=updated_after,
        updated_before=updated_before,
    )
    print(
        dump_json([record.model_dump(mode="json") for record in store.query_runs(query)])
    )
    return 0


@inspect_app.command(name="run-record")
def run_record(*, config: str, run_id: str) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    print(dump_json(get_run_record(store, run_id).model_dump(mode="json")))
    return 0


@inspect_app.command
def lineage(*, config: str, run_id: str | None = None, baseline_label: str | None = None) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    resolved_run_id = resolve_persisted_run_id(
        store,
        run_id=run_id,
        baseline_label=baseline_label,
    )
    record = get_run_record(store, resolved_run_id)
    print(
        dump_json(
            {
                "run_id": record.run_id,
                "lineage": [item.model_dump(mode="json") for item in record.lineage],
            }
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
        message = (
            f"No evaluation execution found for case_id={case_id} metric_id={metric_id}"
        )
        if dataset_id is not None:
            message += f" dataset_id={dataset_id}"
        raise SystemExit(message)
    print(dump_json(execution.model_dump(mode="json")))
    return 0


@inspect_app.command
def case(
    *,
    config: str,
    run_id: str,
    case_id: str,
    dataset_id: str | None = None,
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    audit = get_case_audit(
        store,
        run_id,
        case_id,
        dataset_id=dataset_id,
    )
    if audit is None:
        message = f"No case audit found for case_id={case_id}"
        if dataset_id is not None:
            message += f" dataset_id={dataset_id}"
        raise SystemExit(message)
    print(dump_json(audit.model_dump(mode="json")))
    return 0


@inspect_app.command
def telemetry(*, config: str, run_id: str) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    print(dump_json(get_telemetry_summary(store, run_id).model_dump(mode="json")))
    return 0
