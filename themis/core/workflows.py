"""Evaluation workflow data models for judge-backed metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.models import ConversationTrace, Score, WorkflowTrace
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject

if TYPE_CHECKING:
    from themis.core.contexts import EvalScoreContext


class JudgeCall(HashableModel):
    call_id: str
    judge_model_id: str
    dimension_id: str | None = None
    repeat_index: int = 0
    candidate_indices: list[int] = Field(default_factory=list)
    effective_seed: int | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class RenderedJudgePrompt(HashableModel):
    prompt_id: str
    content: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ParsedJudgment(HashableModel):
    label: str
    score: float | None = None
    rationale: str | None = None
    details: dict[str, JSONValue] = Field(default_factory=dict)


class AggregationResult(HashableModel):
    method: str
    value: JSONValue
    details: dict[str, JSONValue] = Field(default_factory=dict)


class JudgeResponse(HashableModel):
    judge_model_id: str
    judge_model_version: str
    judge_model_fingerprint: str
    effective_seed: int | None = None
    raw_response: str
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None
    provider_request_id: str | None = None
    retry_history: list[dict[str, JSONValue]] = Field(default_factory=list)
    conversation_trace: ConversationTrace | None = None


class WorkflowFailure(HashableModel):
    call_id: str | None = None
    step_id: str
    step_type: str
    error_message: str
    retry_history: list[dict[str, JSONValue]] = Field(default_factory=list)


class EvaluationExecution(HashableModel):
    execution_id: str
    subject_kind: str
    status: str = "completed"
    judge_calls: list[JudgeCall] = Field(default_factory=list)
    rendered_prompts: list[RenderedJudgePrompt] = Field(default_factory=list)
    judge_responses: list[JudgeResponse] = Field(default_factory=list)
    parsed_judgments: list[ParsedJudgment] = Field(default_factory=list)
    scores: list[Score] = Field(default_factory=list)
    failures: list[WorkflowFailure] = Field(default_factory=list)
    aggregation_output: AggregationResult | None = None
    trace: WorkflowTrace


def build_prompt_template_context(
    subject: CandidateSetSubject | TraceSubject | ConversationSubject,
    ctx: EvalScoreContext,
    call: JudgeCall | None = None,
) -> dict[str, JSONValue]:
    candidate_output: JSONValue = ""
    candidate_a_output: JSONValue = ""
    candidate_b_output: JSONValue = ""

    if isinstance(subject, CandidateSetSubject) and subject.candidates:
        candidate_indices = call.candidate_indices if call is not None else []
        if candidate_indices:
            candidate_output = subject.candidates[candidate_indices[0]].final_output
        else:
            candidate_output = subject.candidates[0].final_output

        if len(candidate_indices) >= 2:
            candidate_a_output = subject.candidates[candidate_indices[0]].final_output
            candidate_b_output = subject.candidates[candidate_indices[1]].final_output
        elif len(subject.candidates) >= 2:
            candidate_a_output = subject.candidates[0].final_output
            candidate_b_output = subject.candidates[1].final_output
    elif isinstance(subject, TraceSubject):
        candidate_output = subject.trace.model_dump(mode="json")
    elif isinstance(subject, ConversationSubject):
        candidate_output = subject.conversation.model_dump(mode="json")

    return {
        "candidate_output": candidate_output,
        "candidate_a_output": candidate_a_output,
        "candidate_b_output": candidate_b_output,
        "case_input": ctx.case.input,
        "parsed_output": ctx.parsed_output.value,
    }
