"""Public helpers for inspecting stored execution state."""

from __future__ import annotations

from themis.core.results import ExecutionState
from themis.core.store import RunStore
from themis.core.workflows import EvaluationExecution


def get_execution_state(store: RunStore, run_id: str) -> ExecutionState:
    stored = store.resume(run_id)
    if stored is None:
        raise ValueError(f"Unknown run_id: {run_id}")
    return stored.execution_state


def get_evaluation_execution(
    store: RunStore,
    run_id: str,
    case_id: str,
    metric_id: str,
) -> EvaluationExecution | None:
    state = get_execution_state(store, run_id)
    case_state = state.case_states.get(case_id)
    if case_state is None:
        return None
    return case_state.evaluation_executions.get(metric_id)
