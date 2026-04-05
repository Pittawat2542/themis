from __future__ import annotations

from datetime import UTC, datetime

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
)
from themis.core.prompts import PromptSpec
from themis.core.models import (
    Case,
    ConversationTrace,
    Dataset,
    GenerationResult,
    Message,
    ParsedOutput,
    ReducedCandidate,
    Score,
    ScoreError,
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
        parsed_output=ParsedOutput(value={"answer": "4"}),
        dataset_metadata={"split": "test"},
        seed=7,
    )
    eval_score = EvalScoreContext(
        run_id="run-1",
        case=score.case,
        parsed_output=score.parsed_output,
        dataset_metadata={"split": "test"},
        seed=7,
        judge_model_refs=[
            ComponentRef(
                component_id="builtin/demo_judge",
                version="1.0",
                fingerprint="abc123",
            )
        ],
        judge_seed=9,
        prompt_spec=prompt_spec,
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
    storage = StorageConfig(store="memory", parameters={"path": ":memory:"})

    assert GenerateContext.model_validate_json(generate.model_dump_json()) == generate
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


def test_score_models_capture_success_and_failure() -> None:
    score = Score(metric_id="exact_match", value=1.0, details={"matched": True})
    error = ScoreError(
        metric_id="llm_judge",
        reason="judge timeout",
        retryable=True,
        details={"attempt": 1},
    )
    reduced = ReducedCandidate(
        candidate_id="reduced-1",
        source_candidate_ids=["candidate-1", "candidate-2"],
        final_output="4",
        metadata={"strategy": "majority_vote"},
    )

    assert score.value == 1.0
    assert error.retryable is True
    assert reduced.source_candidate_ids == ["candidate-1", "candidate-2"]
