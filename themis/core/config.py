"""Composed configuration models for Themis v4 experiments."""

from __future__ import annotations

from pydantic import Field

from themis.core.base import HashableModel, JSONValue


class GenerationConfig(HashableModel):
    generator: str
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    reducer: str | None = None


class EvaluationConfig(HashableModel):
    metrics: list[str] = Field(default_factory=list)
    parsers: list[str] = Field(default_factory=list)
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)


class StorageConfig(HashableModel):
    store: str
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
