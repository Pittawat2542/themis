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
    TraceMetric,
)

GeneratorComponent: TypeAlias = Generator | str
SelectorComponent: TypeAlias = CandidateSelector | str
ReducerComponent: TypeAlias = CandidateReducer | str
ParserComponent: TypeAlias = Parser | str
JudgeModelComponent: TypeAlias = JudgeModel | str
MetricComponent: TypeAlias = (
    PureMetric | LLMMetric | SelectionMetric | TraceMetric | str
)


class GenerationConfig(HashableModel):
    """Generation-stage configuration for a run."""

    generator: GeneratorComponent
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    prompt_spec: PromptSpec | None = None
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None


class EvaluationConfig(HashableModel):
    """Evaluation-stage configuration for parsing, metrics, and judges."""

    metrics: list[MetricComponent] = Field(default_factory=list)
    parsers: list[ParserComponent] = Field(default_factory=list)
    judge_models: list[JudgeModelComponent] = Field(default_factory=list)
    prompt_spec: PromptSpec | None = None
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_single_parser(self) -> EvaluationConfig:
        if len(self.parsers) > 1:
            raise ValueError("Themis supports at most one parser")
        return self


class StorageConfig(HashableModel):
    """Store backend configuration used for persistence."""

    store: str
    parameters: dict[str, JSONValue] = Field(default_factory=dict)


class RuntimeConfig(HashableModel):
    """Execution-time controls that do not affect snapshot identity."""

    max_concurrent_tasks: int = 32
    stage_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_rate_limits: dict[str, int] = Field(default_factory=dict)
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
