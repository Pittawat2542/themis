"""Composed configuration models for Themis experiments."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import Field, model_validator

from themis.core.base import HashableModel, JSONValue
from themis.core.prompts import PromptSpec
from themis.core.protocols import (
    CandidateReducer,
    CandidateSelector,
    Generator,
    JudgeModel,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    SessionGenerator,
    TraceMetric,
)


class TargetSpec(HashableModel):
    """Declarative target + kwargs specification for config-driven wiring."""

    target: str
    kwargs: dict[str, JSONValue] = Field(default_factory=dict)


GeneratorComponent: TypeAlias = SessionGenerator | Generator | TargetSpec | str
SelectorComponent: TypeAlias = CandidateSelector | TargetSpec | str
ReducerComponent: TypeAlias = CandidateReducer | TargetSpec | str
ParserComponent: TypeAlias = Parser | TargetSpec | str
JudgeModelComponent: TypeAlias = JudgeModel | TargetSpec | str
MetricComponent: TypeAlias = (
    PureMetric | LLMMetric | SelectionMetric | TraceMetric | TargetSpec | str
)


class ParserView(HashableModel):
    """Named parser view available to metrics during scoring."""

    id: str
    parser: ParserComponent


class SessionConfig(HashableModel):
    """Session-stage configuration for a run."""

    generator: GeneratorComponent
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    prompt_spec: PromptSpec | None = None
    max_turns: int = 1
    termination: dict[str, JSONValue] = Field(default_factory=dict)
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None


class GenerationConfig(SessionConfig):
    """Legacy name for session-stage configuration."""


class EvaluationConfig(HashableModel):
    """Evaluation-stage configuration for parsing, metrics, and judges."""

    metrics: list[MetricComponent] = Field(default_factory=list)
    parsers: list[ParserView | ParserComponent] = Field(default_factory=list)
    judge_models: list[JudgeModelComponent] = Field(default_factory=list)
    prompt_spec: PromptSpec | None = None
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parser_views(self) -> EvaluationConfig:
        seen: set[str] = set()
        for view in self.parser_views:
            if not view.id:
                raise ValueError("Parser view id cannot be empty")
            if view.id in seen:
                raise ValueError(f"Duplicate parser view id: {view.id}")
            seen.add(view.id)
        return self

    @property
    def parser_views(self) -> list[ParserView]:
        if not self.parsers:
            return []
        if len(self.parsers) == 1 and not isinstance(self.parsers[0], ParserView):
            return [ParserView(id="default", parser=self.parsers[0])]
        views: list[ParserView] = []
        for parser in self.parsers:
            if not isinstance(parser, ParserView):
                raise ValueError(
                    "Multiple parsers must be configured as ParserView(id=..., parser=...)"
                )
            views.append(parser)
        return views


class StorageConfig(HashableModel):
    """Store backend configuration used for persistence."""

    target: str = "memory"
    kwargs: dict[str, JSONValue] = Field(default_factory=dict)


class RuntimeConfig(HashableModel):
    """Execution-time controls that do not affect snapshot identity."""

    max_concurrent_tasks: int = 32
    stage_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_rate_limits: dict[str, int] = Field(default_factory=dict)
    provider_token_limits: dict[str, int] = Field(default_factory=dict)
    generation_retry_attempts: int = 3
    generation_retry_delay: float = 0.01
    generation_retry_backoff: float = 2.0
    judge_retry_attempts: int = 3
    judge_retry_delay: float = 0.01
    judge_retry_backoff: float = 2.0
    store_retry_attempts: int = 5
    store_retry_delay: float = 0.01
    existing_run_policy: str = "auto"
    queue_root: str | None = None
    batch_root: str | None = None
