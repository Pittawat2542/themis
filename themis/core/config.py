"""Composed configuration models for Themis v4 experiments."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.protocols import (
    CandidateReducer,
    Generator,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    TraceMetric,
)

GeneratorComponent: TypeAlias = Generator | str
ReducerComponent: TypeAlias = CandidateReducer | str
ParserComponent: TypeAlias = Parser | str
MetricComponent: TypeAlias = PureMetric | LLMMetric | SelectionMetric | TraceMetric | str


class GenerationConfig(HashableModel):
    generator: GeneratorComponent
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    reducer: ReducerComponent | None = None


class EvaluationConfig(HashableModel):
    metrics: list[MetricComponent] = Field(default_factory=list)
    parsers: list[ParserComponent] = Field(default_factory=list)
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)


class StorageConfig(HashableModel):
    store: str
    parameters: dict[str, JSONValue] = Field(default_factory=dict)


class RuntimeConfig(HashableModel):
    max_concurrent_tasks: int = 32
    stage_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_concurrency: dict[str, int] = Field(default_factory=dict)
    provider_rate_limits: dict[str, int] = Field(default_factory=dict)
    store_retry_attempts: int = 5
    store_retry_delay: float = 0.01
