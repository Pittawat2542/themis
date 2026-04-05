from __future__ import annotations

import pytest
from pydantic import ValidationError

from themis.core.events import (
    EvaluationCompletedEvent,
    EvaluationFailedEvent,
    GenerationCompletedEvent,
    GenerationFailedEvent,
    ParseFailedEvent,
    ReductionFailedEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
    ScoreFailedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    StepStartedEvent,
    event_from_dict,
)


def test_run_events_include_schema_version() -> None:
    event = RunStartedEvent(run_id="run-1")

    payload = event.model_dump(mode="json")

    assert payload["schema_version"] == "1"
    assert payload["event_type"] == "run_started"


def test_run_event_round_trip_preserves_subclass() -> None:
    original = StepCompletedEvent(
        run_id="run-1",
        workflow_id="workflow-1",
        step_id="step-1",
        step_type="model_call",
        details={"status": "ok"},
    )

    restored = event_from_dict(original.model_dump(mode="json"))

    assert isinstance(restored, StepCompletedEvent)
    assert restored == original


def test_event_deserialization_supports_all_initial_variants() -> None:
    events: list[RunEvent] = [
        RunStartedEvent(run_id="run-1"),
        RunCompletedEvent(run_id="run-1"),
        RunFailedEvent(run_id="run-1", error_message="boom"),
        GenerationCompletedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-1",
            candidate_index=0,
            seed=7,
            result={"candidate_id": "candidate-1", "final_output": {"answer": "4"}},
            result_blob_ref="sha256:abc123",
        ),
        EvaluationCompletedEvent(
            run_id="run-1",
            case_id="case-1",
            metric_id="metric/judge",
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "trace": {"trace_id": "trace-1", "steps": []},
            },
        ),
        ScoreCompletedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-1",
            metric_id="exact_match",
            score={
                "metric_id": "exact_match",
                "value": 1.0,
                "details": {"matched": True},
            },
        ),
        GenerationFailedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-2",
            error_message="provider timeout",
        ),
        ReductionFailedEvent(
            run_id="run-1",
            case_id="case-1",
            error_message="no candidates",
        ),
        ParseFailedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-1",
            error_message="invalid json",
        ),
        ScoreFailedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-1",
            metric_id="exact_match",
            error={"metric_id": "exact_match", "reason": "missing expected output"},
        ),
        EvaluationFailedEvent(
            run_id="run-1",
            case_id="case-1",
            metric_id="metric/judge",
            error_message="judge unavailable",
        ),
        StepStartedEvent(
            run_id="run-1",
            workflow_id="workflow-1",
            step_id="step-1",
            step_type="render_prompt",
        ),
        StepCompletedEvent(
            run_id="run-1",
            workflow_id="workflow-1",
            step_id="step-1",
            step_type="render_prompt",
        ),
        StepFailedEvent(
            run_id="run-1",
            workflow_id="workflow-1",
            step_id="step-1",
            step_type="render_prompt",
            error_message="failed",
        ),
    ]

    restored = [event_from_dict(event.model_dump(mode="json")) for event in events]

    assert restored == events


def test_known_event_type_accepts_additive_fields_and_preserves_them() -> None:
    payload = {
        "schema_version": "2",
        "event_type": "run_started",
        "run_id": "run-1",
        "new_field": {"added_in": "future"},
    }

    restored = event_from_dict(payload)

    dumped = restored.model_dump(mode="json")

    assert dumped["schema_version"] == "2"
    assert dumped["event_type"] == "run_started"
    assert dumped["run_id"] == "run-1"
    assert dumped["new_field"] == {"added_in": "future"}


def test_known_event_type_accepts_newer_schema_version() -> None:
    restored = event_from_dict(
        {
            "schema_version": "2",
            "event_type": "run_completed",
            "run_id": "run-1",
        }
    )

    assert restored.schema_version == "2"


def test_generation_event_payloads_round_trip_runtime_data() -> None:
    event = GenerationCompletedEvent(
        run_id="run-1",
        case_id="case-1",
        candidate_id="candidate-1",
        candidate_index=1,
        seed=11,
        result={
            "candidate_id": "candidate-1",
            "final_output": {"answer": "4"},
            "conversation": [{"role": "assistant", "content": "4"}],
        },
        result_blob_ref="sha256:payload",
        provider_key="openai:gpt-5.4-mini",
    )

    restored = event_from_dict(event.model_dump(mode="json"))

    assert isinstance(restored, GenerationCompletedEvent)
    assert restored.result_blob_ref == "sha256:payload"
    assert restored.provider_key == "openai:gpt-5.4-mini"
    assert restored.result == event.result


def test_evaluation_event_payloads_round_trip_runtime_data() -> None:
    event = EvaluationCompletedEvent(
        run_id="run-1",
        case_id="case-1",
        metric_id="metric/judge",
        execution={
            "execution_id": "execution-1",
            "subject_kind": "candidate_set",
            "rendered_prompts": [{"prompt_id": "prompt-1", "content": "grade"}],
            "trace": {"trace_id": "trace-1", "steps": []},
        },
        execution_blob_ref="sha256:evaluation",
    )

    restored = event_from_dict(event.model_dump(mode="json"))

    assert isinstance(restored, EvaluationCompletedEvent)
    assert restored.execution_blob_ref == "sha256:evaluation"
    assert restored.execution == event.execution


def test_malformed_known_event_payload_still_fails_validation() -> None:
    with pytest.raises(ValidationError):
        event_from_dict(
            {
                "schema_version": "2",
                "event_type": "run_failed",
                "run_id": "run-1",
            }
        )
