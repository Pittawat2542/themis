"""Experiment-side specification models used by planning and execution."""

from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from themis._replay import ResumeState
from themis.specs.base import SpecBase
from themis.specs.foundational import JudgeInferenceSpec, ModelSpec, TaskSpec
from themis.types.json_types import JSONDict, JSONValueType


class DataItemContext(BaseModel):
    """Execution-side dataset item payload bound to one planned trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(
        ..., description="Unique ID for the specific dataset item/row."
    )
    payload: dict[str, JSONValueType] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> JSONValueType:
        return self.payload[key]

    def get(
        self, key: str, default: JSONValueType | None = None
    ) -> JSONValueType | None:
        """Return one payload value by key, falling back to `default`."""
        return self.payload.get(key, default)

    def keys(self):
        """Return the payload keys exposed to prompt rendering and metrics."""
        return self.payload.keys()

    def items(self):
        """Return the payload items exposed to prompt rendering and metrics."""
        return self.payload.items()


class RuntimeContext(BaseModel):
    """Execution-only runtime inputs and deterministic per-candidate state."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    secrets: dict[str, SecretStr] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    run_labels: dict[str, str] = Field(default_factory=dict)
    candidate_seed: int | None = Field(default=None)
    resume: ResumeState | None = Field(default=None)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return one runtime field by attribute name, with a default fallback."""
        return getattr(self, key, default)


class InferenceParamsSpec(SpecBase):
    """Sampling and response-shape settings forwarded to inference engines."""

    temperature: float = Field(
        default=0.0, ge=0.0, description="Sampling randomness. 0.0 is deterministic."
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold."
    )
    top_k: int | None = Field(default=None, ge=0, description="Top-k token threshold.")
    max_tokens: int = Field(
        default=1024, gt=0, description="Max string length generated."
    )
    stop_sequences: list[str] = Field(
        default_factory=list, description="Sequences that end generation."
    )
    logprobs: int | None = Field(
        default=None, ge=0, description="Request token logprobs if available."
    )
    response_format: Literal["text", "json"] | None = Field(default=None)
    seed: int | None = Field(
        default=None, description="Optional deterministic PRNG seed."
    )
    extras: JSONDict = Field(
        default_factory=dict, description="Provider-specific sampling args."
    )


class PromptMessage(BaseModel):
    """One structured chat message in a prompt template."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class PromptTemplateSpec(SpecBase):
    """Structured chat-style prompt template carried through planning and hooks."""

    id: str | None = Field(
        default=None, description="Optional stable identifier for this template."
    )
    messages: list[PromptMessage] = Field(
        ...,
        description="List of templated messages mimicking standard Chat formats.",
    )


class StorageSpec(SpecBase):
    """Shared storage defaults attached at the project level."""

    backend: Literal["sqlite_blob"] = Field(default="sqlite_blob")
    root_dir: str = Field(
        ..., description="Storage root for event, projection, and blob data."
    )
    store_item_payloads: bool = Field(
        default=True, description="Persist item payload blobs for replayability."
    )
    compression: Literal["none", "zstd"] = Field(default="zstd")


class ExecutionPolicySpec(SpecBase):
    """Retry and circuit-breaker configuration for orchestration."""

    max_retries: int = Field(default=3, ge=0)
    retry_backoff_factor: float = Field(default=1.5, gt=0.0)
    circuit_breaker_threshold: int = Field(default=5, ge=1)


class ItemSamplingSpec(SpecBase):
    """Controls whether a task runs all items or a deterministic subset."""

    kind: Literal["all", "subset", "stratified"] = Field(default="all")
    count: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None)
    strata_field: str | None = Field(
        default=None, description="Field name used for stratified sampling."
    )

    @model_validator(mode="after")
    def _validate_semantic(self) -> ItemSamplingSpec:
        if self.kind in {"subset", "stratified"} and self.count is None:
            raise ValueError(
                f"ItemSamplingSpec kind='{self.kind}' requires a positive count."
            )
        if self.kind == "stratified" and not self.strata_field:
            raise ValueError(
                "ItemSamplingSpec kind='stratified' requires strata_field."
            )
        return self


class InferenceGridSpec(SpecBase):
    """Typed inference sweep over base parameter sets and scalar overrides."""

    params: list[InferenceParamsSpec] = Field(..., min_length=1)
    overrides: dict[str, list[str | int | float | bool]] = Field(default_factory=dict)

    def expand(self) -> list[InferenceParamsSpec]:
        """Expand the base inference params over all configured overrides."""
        if not self.overrides:
            return list(self.params)

        expanded: list[InferenceParamsSpec] = []
        override_keys = sorted(self.overrides)
        override_values = [self.overrides[key] for key in override_keys]

        for base in self.params:
            base_payload = base.model_dump()
            for combination in itertools.product(*override_values):
                payload = dict(base_payload)
                extras = dict(payload.get("extras", {}))
                for key, value in zip(override_keys, combination):
                    if key in InferenceParamsSpec.model_fields:
                        payload[key] = value
                    else:
                        extras[key] = value
                if extras:
                    payload["extras"] = extras
                expanded.append(InferenceParamsSpec.model_validate(payload))
        return expanded


class TrialSpec(SpecBase):
    """Deterministic execution unit for one model/task/item/prompt/params tuple."""

    trial_id: str = Field(..., description="Unique generated deterministic ID.")
    model: ModelSpec
    task: TaskSpec
    item_id: str = Field(
        ..., description="Unique ID for the specific dataset item/row."
    )
    prompt: PromptTemplateSpec
    params: InferenceParamsSpec
    candidate_count: int = Field(
        default=1, ge=1, description="Number of independent candidates to generate."
    )
    metadata: JSONDict = Field(
        default_factory=dict, json_schema_extra={"exclude_from_hash": True}
    )

    @model_validator(mode="after")
    def _validate_semantic(self) -> TrialSpec:
        if self.candidate_count < 1:
            raise ValueError("Candidate count must be >= 1")
        return self


class ExperimentSpec(SpecBase):
    """
    Author-facing experiment matrix expanded into `TrialSpec` objects by the planner.
    """

    models: list[ModelSpec] = Field(..., description="Models to test.")
    tasks: list[TaskSpec] = Field(..., description="Evaluation tasks to run against.")
    prompt_templates: list[PromptTemplateSpec] = Field(
        ..., description="Prompt templates to multiplex."
    )
    inference_grid: InferenceGridSpec = Field(
        ..., description="Typed sampling grid to cartesian product."
    )
    num_samples: int = Field(
        default=1, ge=1, description="How many samples per trial to draw by default."
    )
    item_sampling: ItemSamplingSpec = Field(default_factory=ItemSamplingSpec)

    @model_validator(mode="after")
    def _validate_semantic(self) -> ExperimentSpec:
        if not self.models:
            raise ValueError("ExperimentSpec must have at least one model.")
        if not self.tasks:
            raise ValueError("ExperimentSpec must have at least one task.")
        if not self.prompt_templates:
            raise ValueError("ExperimentSpec must have at least one prompt template.")
        if self.num_samples < 1:
            raise ValueError("ExperimentSpec num_samples must be >= 1.")
        return self


class ProjectSpec(SpecBase):
    """Shared project-level identity, storage defaults, and execution policy."""

    project_name: str = Field(..., description="Human readable project name.")
    researcher_id: str = Field(
        ..., description="Stable owner identifier for experiment lineage."
    )
    global_seed: int = Field(
        ..., description="Default deterministic seed shared across experiments."
    )
    storage: StorageSpec = Field(..., description="Shared storage defaults.")
    execution_policy: ExecutionPolicySpec = Field(
        ..., description="Shared retry and circuit-breaker policy."
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="User-defined project metadata."
    )


JudgeInferenceSpec.model_rebuild(
    _types_namespace={"InferenceParamsSpec": InferenceParamsSpec}
)
