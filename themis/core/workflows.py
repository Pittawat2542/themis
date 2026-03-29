"""Evaluation workflow data models for judge-backed metrics."""

from __future__ import annotations

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.models import ConversationTrace, Score, WorkflowTrace


class EvalStep(HashableModel):
    step_type: str
    config: dict[str, JSONValue] = Field(default_factory=dict)


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


class EvaluationExecution(HashableModel):
    execution_id: str
    subject_kind: str
    rendered_prompts: list[RenderedJudgePrompt] = Field(default_factory=list)
    judge_responses: list[JudgeResponse] = Field(default_factory=list)
    parsed_judgments: list[ParsedJudgment] = Field(default_factory=list)
    scores: list[Score] = Field(default_factory=list)
    aggregation_output: AggregationResult | None = None
    trace: WorkflowTrace
