"""Typed immutable context models passed across execution stages."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.components import ComponentRef
from themis.core.models import Case, ParsedOutput
from themis.core.prompts import PromptSpec


class GenerateContext(HashableModel):
    """Context passed to generators for one case execution."""

    run_id: str
    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    seed: int | None = None
    prompt_spec: PromptSpec | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SelectContext(HashableModel):
    """Context passed to candidate selectors before reduction."""

    run_id: str
    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    judge_models: list[Any] = Field(default_factory=list, exclude=True)


class ReduceContext(HashableModel):
    """Context passed to reducers choosing a final candidate."""

    run_id: str
    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ParseContext(HashableModel):
    """Context passed to parsers for a reduced candidate."""

    run_id: str
    case_id: str
    dataset_id: str | None = None
    case_key: str | None = None
    candidate_id: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ScoreContext(HashableModel):
    """Context passed to deterministic scoring metrics."""

    run_id: str
    case: Case
    parsed_output: ParsedOutput
    dataset_id: str | None = None
    case_key: str | None = None
    dataset_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    seed: int | None = None


class EvalScoreContext(ScoreContext):
    """Score context extended with judge workflow configuration."""

    judge_model_refs: list[ComponentRef] = Field(default_factory=list)
    judge_seed: int | None = None
    prompt_spec: PromptSpec | None = None
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    eval_workflow_config: dict[str, JSONValue] = Field(default_factory=dict)
