from __future__ import annotations

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    GenerationCompletedEvent,
    GenerationFailedEvent,
    ParseCompletedEvent,
    ReductionCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
)
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.projections import (
    build_benchmark_result,
    build_run_result,
    build_timeline_view,
    build_trace_view,
)
from themis.core.workflows import EvaluationExecution


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7, 11],
        environment_metadata={"env": "test"},
        themis_version="4.0.0rc1",
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


def _events(run_id: str):
    return [
        RunStartedEvent(run_id=run_id),
        GenerationCompletedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="candidate-1",
            candidate_index=0,
            seed=7,
            result={
                "candidate_id": "candidate-1",
                "final_output": {"answer": "4"},
                "trace": [
                    {
                        "step_name": "draft",
                        "step_type": "model_call",
                        "output": {"answer": "4"},
                    }
                ],
                "conversation": [{"role": "assistant", "content": "4"}],
            },
            result_blob_ref="sha256:generation-1",
        ),
        GenerationFailedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="candidate-2",
            error_message="provider timeout",
        ),
        ReductionCompletedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            source_candidate_ids=["candidate-1"],
            result={
                "candidate_id": "case-1-reduced",
                "source_candidate_ids": ["candidate-1"],
                "final_output": {"answer": "4"},
                "metadata": {"strategy": "first_candidate"},
            },
        ),
        ParseCompletedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            result={"value": {"answer": "4"}, "format": "json"},
        ),
        EvaluationCompletedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "judge_calls": [
                    {"call_id": "call-1", "judge_model_id": "builtin/demo_judge"}
                ],
                "rendered_prompts": [{"prompt_id": "prompt-1", "content": "grade"}],
                "judge_responses": [
                    {
                        "judge_model_id": "builtin/demo_judge",
                        "judge_model_version": "1.0",
                        "judge_model_fingerprint": "builtin-judge-demo-fingerprint",
                        "raw_response": "pass",
                    }
                ],
                "parsed_judgments": [{"label": "pass", "score": 1.0}],
                "scores": [{"metric_id": "metric/judge", "value": 1.0}],
                "aggregation_output": {
                    "method": "mean",
                    "value": 1.0,
                    "details": {"votes": 1},
                },
                "trace": {"trace_id": "trace-1", "steps": []},
            },
            execution_blob_ref="sha256:evaluation-1",
        ),
        EvaluationFailedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/trace",
            error_message="judge unavailable",
        ),
        ScoreCompletedEvent(
            run_id=run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="builtin/exact_match",
            score={
                "metric_id": "builtin/exact_match",
                "value": 1.0,
                "details": {"matched": True},
            },
        ),
        RunCompletedEvent(run_id=run_id),
    ]


def test_build_run_result_projects_case_drill_down_from_events() -> None:
    snapshot = _snapshot()

    result = build_run_result(snapshot, _events(snapshot.run_id))

    assert result.run_id == snapshot.run_id
    assert result.status.value == "partial_failure"
    assert result.progress.total_cases == 1
    assert result.progress.completed_cases == 0
    assert result.progress.failed_cases == 1
    assert len(result.cases) == 1

    case = result.cases[0]
    assert case.case_id == "case-1"
    assert case.generated_candidates[0].candidate_id == "candidate-1"
    assert case.generated_candidates[0].trace is not None
    assert case.generated_candidates[0].conversation is not None
    assert case.generated_candidate_blob_refs == {"candidate-1": "sha256:generation-1"}
    assert case.generation_failures == {"candidate-2": "provider timeout"}
    assert case.reduced_candidate is not None
    assert case.parsed_output is not None
    assert case.evaluation_executions[0].execution_id == "execution-1"
    assert case.evaluation_execution_blob_refs == {
        "metric/judge": "sha256:evaluation-1"
    }
    assert case.evaluation_failures == {"metric/trace": "judge unavailable"}
    assert case.scores[0].metric_id == "builtin/exact_match"


def test_build_benchmark_result_aggregates_scores_from_run_result() -> None:
    snapshot = _snapshot()

    result = build_benchmark_result(snapshot, _events(snapshot.run_id))

    assert result.run_id == snapshot.run_id
    assert result.dataset_ids == ["dataset-1"]
    assert result.metric_ids == ["builtin/exact_match"]
    assert result.total_cases == 1
    assert result.completed_cases == 0
    assert result.failed_cases == 1
    assert len(result.score_rows) == 1
    assert result.score_rows[0].case_id == "case-1"
    assert result.score_rows[0].metric_id == "builtin/exact_match"
    assert result.score_rows[0].value == 1.0
    assert result.metric_means == {"builtin/exact_match": 1.0}


def test_build_timeline_view_preserves_event_order_and_case_scope() -> None:
    snapshot = _snapshot()

    view = build_timeline_view(snapshot, _events(snapshot.run_id))

    assert [entry.event_type for entry in view.entries] == [
        "run_started",
        "generation_completed",
        "generation_failed",
        "reduction_completed",
        "parse_completed",
        "evaluation_completed",
        "evaluation_failed",
        "score_completed",
        "run_completed",
    ]
    assert view.entries[1].case_id == "case-1"
    assert view.entries[1].candidate_id == "candidate-1"
    assert view.entries[5].metric_id == "metric/judge"


def test_build_trace_view_collects_generation_and_evaluation_traces() -> None:
    snapshot = _snapshot()

    view = build_trace_view(snapshot, _events(snapshot.run_id))

    assert len(view.generation_traces) == 1
    assert view.generation_traces[0].case_id == "case-1"
    assert view.generation_traces[0].candidate_id == "candidate-1"
    assert view.generation_traces[0].trace_id == "candidate-1:generation"
    assert len(view.generation_traces[0].steps) == 1

    assert len(view.conversation_traces) == 1
    assert view.conversation_traces[0].candidate_id == "candidate-1"
    assert view.conversation_traces[0].trace_id == "candidate-1:conversation"

    assert len(view.evaluation_traces) == 1
    assert view.evaluation_traces[0].case_id == "case-1"
    assert view.evaluation_traces[0].metric_id == "metric/judge"
    assert isinstance(view.evaluation_traces[0].execution, EvaluationExecution)


def test_build_run_result_marks_partial_workflow_execution_failures() -> None:
    snapshot = _snapshot()
    events = _events(snapshot.run_id)
    events[5] = EvaluationCompletedEvent(
        run_id=snapshot.run_id,
        case_id="case-1",
        candidate_id="case-1-reduced",
        metric_id="metric/judge",
        execution={
            "execution_id": "execution-1",
            "subject_kind": "candidate_set",
            "scores": [{"metric_id": "metric/judge", "value": 1.0}],
            "failures": [
                {
                    "call_id": "call-2",
                    "step_id": "call-2:model_call",
                    "step_type": "model_call",
                    "error_message": "judge timeout",
                }
            ],
            "status": "partial_failure",
            "trace": {"trace_id": "trace-1", "steps": []},
        },
    )

    result = build_run_result(snapshot, events)

    assert result.status.value == "partial_failure"
    execution = result.cases[0].evaluation_executions[0]
    assert execution.status == "partial_failure"
    assert execution.failures[0].call_id == "call-2"
