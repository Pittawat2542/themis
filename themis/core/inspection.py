"""Public helpers for inspecting stored execution state."""

from __future__ import annotations

from themis.core.case_refs import resolve_case_key
from themis.core.events import (
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    EvaluationCompletedEvent,
    ScoreCompletedEvent,
)
from themis.core.read_models import AttemptSummary, CaseAuditView, TelemetrySummary
from themis.core.registry import RunQuery, RunRecord
from themis.core.results import ExecutionState
from themis.core.snapshot import RunSnapshot
from themis.core.store import ProjectionConsistency, ProjectionRead, RunStore
from themis.core.workflows import EvaluationExecution


def get_run_snapshot(store: RunStore, run_id: str) -> RunSnapshot:
    """Return the persisted snapshot for a run."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return stored.snapshot


def get_projection(
    store: RunStore,
    run_id: str,
    projection_name: str,
    *,
    consistency: ProjectionConsistency = ProjectionConsistency.FRESH,
) -> ProjectionRead:
    """Read a projection with explicit freshness semantics."""

    return store.read_projection(run_id, projection_name, consistency=consistency)


def get_attempt_history(store: RunStore, run_id: str) -> list[AttemptSummary]:
    """Return lifecycle and score-claim history for every stored attempt."""

    attempts: dict[str, AttemptSummary] = {}
    for event in store.query_events(run_id):
        attempt_id = event.attempt_id or "legacy"
        current = attempts.get(attempt_id, AttemptSummary(attempt_id=attempt_id))
        if isinstance(event, RunStartedEvent):
            current = current.model_copy(
                update={
                    "attempt_kind": event.attempt_kind,
                    "parent_attempt_id": event.parent_attempt_id,
                    "status": "running",
                    "started_at": event.occurred_at,
                }
            )
        elif isinstance(event, RunCompletedEvent):
            current = current.model_copy(
                update={"status": "completed", "ended_at": event.occurred_at}
            )
        elif isinstance(event, RunFailedEvent):
            current = current.model_copy(
                update={"status": "failed", "ended_at": event.occurred_at}
            )
        if isinstance(event, ScoreCompletedEvent | EvaluationCompletedEvent):
            current = current.model_copy(
                update={"score_claim_count": current.score_claim_count + 1}
            )
        attempts[attempt_id] = current
    return list(attempts.values())


def get_score_claim_history(
    store: RunStore,
    run_id: str,
    *,
    metric_id: str | None = None,
) -> list[ScoreCompletedEvent]:
    """Return historical pure-metric claims without collapsing attempts."""

    return [
        event
        for event in store.query_events(run_id)
        if isinstance(event, ScoreCompletedEvent)
        and (metric_id is None or event.metric_id == metric_id)
    ]


def get_execution_state(store: RunStore, run_id: str) -> ExecutionState:
    """Return the persisted execution state for a run."""

    checkpoint = store.load_execution_checkpoint(run_id)
    event_count = store.count_events(run_id)
    if checkpoint is not None and checkpoint.event_count == event_count:
        return checkpoint.execution_state
    return _require_stored_run(store, run_id).execution_state


def get_evaluation_execution(
    store: RunStore,
    run_id: str,
    case_id: str,
    metric_id: str,
    *,
    dataset_id: str | None = None,
    case_key: str | None = None,
) -> EvaluationExecution | None:
    """Return one stored workflow execution for a case and metric."""

    stored = _require_stored_run(store, run_id)
    resolved_case_key = _resolve_stored_case_key(
        stored.snapshot,
        case_id=case_id,
        dataset_id=dataset_id,
        case_key=case_key,
    )
    case_state = stored.execution_state.case_states.get(resolved_case_key)
    if case_state is None and dataset_id is None and case_key is None:
        case_state = stored.execution_state.case_states.get(case_id)
    if case_state is None:
        return None
    return case_state.evaluation_executions.get(metric_id)


def get_case_audit(
    store: RunStore,
    run_id: str,
    case_id: str,
    *,
    dataset_id: str | None = None,
    case_key: str | None = None,
):
    """Return the unified pipeline audit record for one case."""

    projection = store.get_projection(run_id, "case_audit_view")
    if not isinstance(projection, dict):
        raise ValueError(f"Case audit projection unavailable for run_id={run_id}")
    audit_view = CaseAuditView.model_validate(projection)
    resolved_case_key = resolve_case_key(
        case_id=case_id,
        dataset_id=dataset_id,
        case_key=case_key,
    )
    for record in audit_view.cases:
        if record.case_key == resolved_case_key or (
            record.case_id == case_id
            and (dataset_id is None or record.dataset_id == dataset_id)
        ):
            return record
    return None


def get_telemetry_summary(store: RunStore, run_id: str) -> TelemetrySummary:
    """Return the aggregated telemetry summary for a run."""

    projection = store.get_projection(run_id, "telemetry_summary")
    if not isinstance(projection, dict):
        raise ValueError(f"Telemetry summary unavailable for run_id={run_id}")
    return TelemetrySummary.model_validate(projection)


def get_run_record(store: RunStore, run_id: str) -> RunRecord:
    """Return the registry record for one persisted run."""

    record = store.get_run_record(run_id)
    if record is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return record


def query_run_records(
    store: RunStore, query: RunQuery | None = None
) -> list[RunRecord]:
    """Query registry records from a run store."""

    return store.query_runs(query)


def resolve_run_record(
    store: RunStore,
    *,
    run_id: str | None = None,
    baseline_label: str | None = None,
    query: RunQuery | None = None,
) -> RunRecord:
    """Resolve a single run record by explicit id or registry query."""

    if run_id is not None:
        return get_run_record(store, run_id)
    resolved_query = query or RunQuery()
    if baseline_label is not None:
        resolved_query = resolved_query.model_copy(
            update={"baseline_label": baseline_label}
        )
    records = query_run_records(store, resolved_query)
    if not records:
        raise ValueError("No stored run matched the provided registry lookup")
    if len(records) > 1:
        raise ValueError("Multiple stored runs matched the provided registry lookup")
    return records[0]


def resolve_run_id(
    store: RunStore,
    *,
    run_id: str | None = None,
    baseline_label: str | None = None,
    query: RunQuery | None = None,
) -> str:
    """Resolve one persisted run id by explicit id or registry query."""

    return resolve_run_record(
        store, run_id=run_id, baseline_label=baseline_label, query=query
    ).run_id


def _resolve_stored_case_key(
    snapshot: RunSnapshot,
    *,
    case_id: str,
    dataset_id: str | None = None,
    case_key: str | None = None,
) -> str:
    if dataset_id is not None or case_key is not None:
        resolved_case_key = resolve_case_key(
            case_id=case_id,
            dataset_id=dataset_id,
            case_key=case_key,
        )
        resolved_case_id = _snapshot_case_id_for_key(snapshot, resolved_case_key)
        if resolved_case_id is not None and resolved_case_id != case_id:
            raise ValueError(
                "Conflicting case_id and case_key inputs: "
                f"case_id={case_id} case_key={resolved_case_key}"
            )
        return resolved_case_key

    matches = [
        resolve_case_key(case_id=case.case_id, dataset_id=dataset.dataset_id)
        for dataset in snapshot.datasets
        for case in dataset.cases
        if case.case_id == case_id
    ]
    if len(matches) == 1:
        return matches[0]
    return case_id


def _snapshot_case_id_for_key(snapshot: RunSnapshot, case_key: str) -> str | None:
    for dataset in snapshot.datasets:
        for case in dataset.cases:
            if (
                resolve_case_key(case_id=case.case_id, dataset_id=dataset.dataset_id)
                == case_key
            ):
                return case.case_id
    return None


def _require_stored_run(store: RunStore, run_id: str):
    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return stored
