"""Public helpers for inspecting stored execution state."""

from __future__ import annotations

from themis.core.case_refs import resolve_case_key
from themis.core.results import ExecutionState
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.workflows import EvaluationExecution


def get_run_snapshot(store: RunStore, run_id: str) -> RunSnapshot:
    """Return the persisted snapshot for a run."""

    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return stored.snapshot


def get_execution_state(store: RunStore, run_id: str) -> ExecutionState:
    """Return the persisted execution state for a run."""

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

    state = get_execution_state(store, run_id)
    resolved_case_key = resolve_case_key(
        case_id=case_id, dataset_id=dataset_id, case_key=case_key
    )
    case_state = state.case_states.get(resolved_case_key, state.case_states.get(case_id))
    if case_state is None:
        return None
    return case_state.evaluation_executions.get(metric_id)


def _require_stored_run(store: RunStore, run_id: str):
    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return stored
