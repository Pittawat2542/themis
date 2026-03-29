from __future__ import annotations

from themis.core.events import (
    EvaluationCompletedEvent,
    GenerationCompletedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
)
from themis.core.results import ExecutionState, RunStatus
from themis.core.snapshot import StoredRun
from themis.core.experiment import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
    )
    return experiment.compile()


def test_execution_state_reconstructs_completed_pipeline_from_events() -> None:
    state = ExecutionState.from_events(
        "run-1",
        [
            RunStartedEvent(run_id="run-1"),
            GenerationCompletedEvent(
                run_id="run-1",
                case_id="case-1",
                candidate_id="candidate-1",
                result={"candidate_id": "candidate-1", "final_output": {"answer": "4"}},
            ),
            ReductionCompletedEvent(
                run_id="run-1",
                case_id="case-1",
                candidate_id="case-1-reduced",
                source_candidate_ids=["candidate-1"],
                result={
                    "candidate_id": "case-1-reduced",
                    "source_candidate_ids": ["candidate-1"],
                    "final_output": {"answer": "4"},
                },
            ),
            ParseCompletedEvent(
                run_id="run-1",
                case_id="case-1",
                candidate_id="case-1-reduced",
                result={"value": {"answer": "4"}, "format": "json"},
            ),
            EvaluationCompletedEvent(
                run_id="run-1",
                case_id="case-1",
                metric_id="metric/judge",
                execution={
                    "execution_id": "execution-1",
                    "subject_kind": "candidate_set",
                    "scores": [{"metric_id": "metric/judge", "value": 1.0}],
                    "trace": {"trace_id": "trace-1", "steps": []},
                },
            ),
            ScoreCompletedEvent(
                run_id="run-1",
                case_id="case-1",
                candidate_id="case-1-reduced",
                metric_id="metric/demo",
                score={"metric_id": "metric/demo", "value": 1.0, "details": {"matched": True}},
            ),
            RunCompletedEvent(run_id="run-1"),
        ],
    )

    assert state.status is RunStatus.COMPLETED
    assert state.case_states["case-1"].generated_candidates["candidate-1"].final_output == {"answer": "4"}
    assert state.case_states["case-1"].reduced_candidate is not None
    assert state.case_states["case-1"].parsed_output is not None
    assert state.case_states["case-1"].evaluation_executions["metric/judge"].execution_id == "execution-1"
    assert state.case_states["case-1"].successful_scores["metric/demo"].value == 1.0


def test_stored_run_exposes_execution_state() -> None:
    snapshot = _snapshot()
    stored = StoredRun(
        snapshot=snapshot,
        events=[
            RunStartedEvent(run_id=snapshot.run_id),
            RunCompletedEvent(run_id=snapshot.run_id),
        ],
    )

    assert stored.execution_state.status is RunStatus.COMPLETED
