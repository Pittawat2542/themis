from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SessionContext,
)
from themis.core.prompts import PromptSpec
from themis.core.models import (
    Case,
    ConversationTrace,
    Dataset,
    FailureCategory,
    GenerationResult,
    MetricResult,
    Message,
    ParsedOutput,
    ReducedCandidate,
    ScoreError,
    SessionResult,
    SessionTurn,
    StreamEvent,
    TraceStep,
    WorkflowTrace,
)
from themis.core.snapshot import ComponentRef


def test_core_models_are_frozen() -> None:
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="4",
        metadata={"source": "unit"},
    )

    with pytest.raises(Exception):
        case.case_id = "mutated"  # type: ignore[misc]


def test_core_models_round_trip_json() -> None:
    result = GenerationResult(
        candidate_id="candidate-1",
        final_output={"answer": "4"},
        trace=[
            TraceStep(
                step_name="model",
                step_type="model_call",
                input={"prompt": "2+2"},
                output={"text": "4"},
                timestamp=datetime(2026, 3, 29, 10, 0, tzinfo=UTC),
            )
        ],
        conversation=[Message(role="assistant", content="4")],
        artifacts={"raw": "4"},
        token_usage={"prompt_tokens": 4, "completion_tokens": 1},
        latency_ms=12.5,
    )

    restored = GenerationResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_session_models_round_trip_json_and_hash_stably() -> None:
    stream_event = StreamEvent(
        event_id="event-1",
        source_stage="session",
        event_type="token",
        payload={"text": "4"},
        timestamp=datetime(2026, 3, 29, 10, 0, tzinfo=UTC),
        offset_ms=12.5,
    )
    session = SessionResult(
        candidate_id="candidate-1",
        final_output={"answer": "4"},
        turns=[
            SessionTurn(
                turn_index=0,
                input_messages=[Message(role="user", content="2+2")],
                output_messages=[Message(role="assistant", content="4")],
                artifacts={"tool": "calculator"},
                trace=[
                    TraceStep(
                        step_name="solve",
                        step_type="reasoning",
                        output={"answer": "4"},
                    )
                ],
                latency_ms=20.0,
            )
        ],
        stream_events=[stream_event],
        environment_state={"terminated": True},
        termination_reason="completed",
        token_usage={"prompt_tokens": 4, "completion_tokens": 1},
        latency_ms=21.0,
    )

    restored = SessionResult.model_validate_json(session.model_dump_json())

    assert restored == session
    assert restored.compute_hash() == session.compute_hash()
    assert restored.stream_events[0].payload == {"text": "4"}
    assert restored.turns[0].output_messages[0].content == "4"


def test_canonical_hashing_is_stable_for_core_models() -> None:
    left = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="4",
        metadata={"source": "unit"},
    )
    right = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="4",
        metadata={"source": "unit"},
    )
    changed = Case(
        case_id="case-1",
        input={"question": "3+3"},
        expected_output="6",
        metadata={"source": "unit"},
    )

    assert left.compute_hash() == right.compute_hash()
    assert left.compute_hash() != changed.compute_hash()


def test_canonical_hashing_recomputes_after_nested_mutation() -> None:
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output={"answer": "4"},
    )

    initial_hash = case.compute_hash()
    cast(dict[str, JSONValue], case.input)["question"] = "3+3"

    assert case.compute_hash() != initial_hash


def test_dataset_and_trace_models_embed_core_records() -> None:
    dataset = Dataset(
        dataset_id="dataset-1",
        cases=[
            Case(
                case_id="case-1",
                input={"question": "2+2"},
                expected_output="4",
            )
        ],
        revision="r1",
        metadata={"owner": "tests"},
    )
    workflow_trace = WorkflowTrace(
        trace_id="trace-1",
        steps=[
            TraceStep(
                step_name="judge",
                step_type="model_call",
                input={"prompt": "grade"},
                output={"score": 1},
            )
        ],
    )
    conversation_trace = ConversationTrace(
        trace_id="conversation-1",
        messages=[Message(role="user", content="Hello")],
    )

    assert dataset.revision == "r1"
    assert workflow_trace.steps[0].step_name == "judge"
    assert conversation_trace.messages[0].role == "user"


def test_contexts_and_configs_serialize_cleanly() -> None:
    prompt_spec = PromptSpec(
        instructions="Answer directly.",
        blocks=[
            {
                "title": "Reference pair",
                "input": {"question": "1+1"},
                "output": {"answer": "2"},
            }
        ],
    )
    generate = GenerateContext(
        run_id="run-1", case_id="case-1", seed=7, prompt_spec=prompt_spec
    )
    session = SessionContext(
        run_id="run-1",
        case_id="case-1",
        seed=7,
        prompt_spec=prompt_spec,
        max_turns=3,
        metadata={"mode": "interactive"},
    )
    reduce = ReduceContext(
        run_id="run-1",
        case_id="case-1",
        candidate_ids=["candidate-1", "candidate-2"],
        seed=7,
    )
    parse = ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1")
    score = ScoreContext(
        run_id="run-1",
        case=Case(case_id="case-1", input={"question": "2+2"}, expected_output="4"),
        parsed_views={"default": ParsedOutput(value={"answer": "4"})},
        parser_view="default",
        dataset_metadata={"split": "test"},
        seed=7,
    )
    eval_score = EvalScoreContext(
        run_id="run-1",
        case=score.case,
        parsed_views=score.parsed_views,
        parser_view=score.parser_view,
        dataset_metadata={"split": "test"},
        seed=7,
        dataset_id="dataset-1",
        case_key="dataset-1:case-1",
        judge_model_refs=[
            ComponentRef(
                component_id="builtin/demo_judge",
                version="1.0",
                fingerprint="abc123",
            )
        ],
        judge_seed=9,
        prompt_spec=prompt_spec,
        judge_config={"panel_size": 1},
        eval_workflow_config={"rubric": "pass_fail"},
    )
    generation = GenerationConfig(
        generator="demo-generator",
        candidate_policy={"num_samples": 2},
        prompt_spec=prompt_spec,
        reducer="demo-reducer",
    )
    evaluation = EvaluationConfig(
        metrics=["exact_match"],
        parsers=["json"],
        prompt_spec=prompt_spec,
        judge_config={"panel_size": 1},
    )
    storage = StorageConfig(target="memory", kwargs={"path": ":memory:"})

    assert GenerateContext.model_validate_json(generate.model_dump_json()) == generate
    assert SessionContext.model_validate_json(session.model_dump_json()) == session
    assert ReduceContext.model_validate_json(reduce.model_dump_json()) == reduce
    assert ParseContext.model_validate_json(parse.model_dump_json()) == parse
    assert ScoreContext.model_validate_json(score.model_dump_json()) == score
    assert (
        EvalScoreContext.model_validate_json(eval_score.model_dump_json()) == eval_score
    )
    assert isinstance(eval_score.judge_model_refs[0], ComponentRef)
    assert (
        GenerationConfig.model_validate_json(generation.model_dump_json()) == generation
    )
    assert (
        EvaluationConfig.model_validate_json(evaluation.model_dump_json()) == evaluation
    )
    assert StorageConfig.model_validate_json(storage.model_dump_json()) == storage


def test_metric_result_models_capture_rich_success_and_failure() -> None:
    result = MetricResult(
        metric_id="exact_match",
        result_type="scalar",
        value=1.0,
        dimensions={"precision": 1.0},
        labels={"winner": "candidate-a"},
        confidence=0.95,
        metadata={"matched": True},
    )
    error = ScoreError(
        metric_id="llm_judge",
        reason="judge timeout",
        retryable=True,
        category=FailureCategory.PROVIDER_FAILURE,
        metadata={"attempt": 1},
    )
    reduced = ReducedCandidate(
        candidate_id="reduced-1",
        source_candidate_ids=["candidate-1", "candidate-2"],
        final_output="4",
        metadata={"strategy": "majority_vote"},
    )

    assert result.value == 1.0
    assert result.result_type == "scalar"
    assert result.dimensions == {"precision": 1.0}
    assert result.labels == {"winner": "candidate-a"}
    assert result.confidence == 0.95
    assert result.metadata == {"matched": True}
    assert error.category is FailureCategory.PROVIDER_FAILURE
    assert error.retryable is True
    assert reduced.source_candidate_ids == ["candidate-1", "candidate-2"]
