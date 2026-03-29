from __future__ import annotations

from themis.core.events import (
    GenerationCompletedEvent,
    RunCompletedEvent,
    RunEvent,
    RunFailedEvent,
    RunStartedEvent,
    ScoreCompletedEvent,
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
        ),
        ScoreCompletedEvent(
            run_id="run-1",
            case_id="case-1",
            candidate_id="candidate-1",
            metric_id="exact_match",
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
