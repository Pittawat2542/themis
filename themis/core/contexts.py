"""Typed immutable context models passed across execution stages."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from themis.core.base import HashableModel, JSONValue
from themis.core.components import ComponentRef
from themis.core.models import Case, ParsedOutput
from themis.core.prompts import PromptSpec


class GenerateContext(HashableModel):
    run_id: str
    case_id: str
    seed: int | None = None
    prompt_spec: PromptSpec | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SelectContext(HashableModel):
    run_id: str
    case_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    judge_models: list[Any] = Field(default_factory=list, exclude=True)


class ReduceContext(HashableModel):
    run_id: str
    case_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ParseContext(HashableModel):
    run_id: str
    case_id: str
    candidate_id: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class ScoreContext(HashableModel):
    run_id: str
    case: Case
    parsed_output: ParsedOutput
    dataset_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    seed: int | None = None


class EvalScoreContext(ScoreContext):
    judge_model_refs: list[ComponentRef] = Field(default_factory=list)
    judge_seed: int | None = None
    prompt_spec: PromptSpec | None = None
    eval_workflow_config: dict[str, JSONValue] = Field(default_factory=dict)
