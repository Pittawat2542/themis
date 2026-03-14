"""Experiment-side specification models used by planning and execution."""

from __future__ import annotations

import itertools
from typing import Annotated
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from themis._replay import ResumeState
from themis.specs.base import SpecBase
from themis.specs.foundational import JudgeInferenceSpec, ModelSpec, TaskSpec
from themis.types.enums import (
    CompressionCodec,
    ResponseFormat,
    SamplingKind,
    StorageBackend,
)
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
    response_format: ResponseFormat | None = Field(default=None)
    seed: int | None = Field(
        default=None, description="Optional deterministic PRNG seed."
    )
    extras: JSONDict = Field(
        default_factory=dict, description="Provider-specific sampling args."
    )

    @field_validator("response_format", mode="before")
    @classmethod
    def _coerce_response_format(
        cls, value: ResponseFormat | str | None
    ) -> ResponseFormat | str | None:
        if isinstance(value, str):
            return ResponseFormat(value)
        return value


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


class _StorageSpecBase(SpecBase):
    """Shared storage defaults attached at the project level."""

    backend: StorageBackend
    store_item_payloads: bool = Field(
        default=True, description="Persist item payload blobs for replayability."
    )
    compression: CompressionCodec = Field(default=CompressionCodec.ZSTD)

    @field_validator("compression", mode="before")
    @classmethod
    def _coerce_compression(
        cls, value: CompressionCodec | str
    ) -> CompressionCodec | str:
        if isinstance(value, str):
            return CompressionCodec(value)
        return value


class SqliteBlobStorageSpec(_StorageSpecBase):
    """SQLite event/projection store plus local filesystem blob persistence."""

    backend: Literal[StorageBackend.SQLITE_BLOB] = Field(
        default=StorageBackend.SQLITE_BLOB
    )
    root_dir: str = Field(
        ..., description="Storage root for event, projection, and blob data."
    )


class PostgresBlobStorageSpec(_StorageSpecBase):
    """Postgres event/projection store plus local filesystem blob persistence."""

    backend: Literal[StorageBackend.POSTGRES_BLOB] = Field(
        default=StorageBackend.POSTGRES_BLOB
    )
    database_url: str = Field(
        ..., description="Postgres connection URL for events and projections."
    )
    blob_root_dir: str = Field(
        ..., description="Local blob root for content-addressed artifact storage."
    )


StorageConfig = Annotated[
    SqliteBlobStorageSpec | PostgresBlobStorageSpec,
    Field(discriminator="backend"),
]

# Keep the SQLite config as the simple top-level convenience constructor.
StorageSpec = SqliteBlobStorageSpec


class ExecutionPolicySpec(SpecBase):
    """Retry, backoff, circuit-breaker, and concurrency controls for orchestration.

    These settings live above provider SDK behavior. Engines are still
    responsible for classifying provider failures into stable retryable codes.
    """

    max_retries: int = Field(default=3, ge=0)
    retry_backoff_factor: float = Field(default=1.5, gt=0.0)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    max_in_flight_work_items: int = Field(default=32, ge=1)
    retryable_error_codes: list[str] = Field(
        default_factory=list,
        description="Stable error-code values treated as retryable for persisted work items.",
    )


class LocalExecutionBackendSpec(SpecBase):
    """Default in-process backend for synchronous local runs."""

    kind: Literal["local"] = Field(default="local")


class WorkerPoolExecutionBackendSpec(SpecBase):
    """Shared-store backend for externally operated worker pools.

    Themis persists run manifests, work items, and lease metadata; external
    workers are responsible for actually executing those items.
    """

    kind: Literal["worker_pool"] = Field(default="worker_pool")
    lease_ttl_seconds: int = Field(default=180, ge=1)
    poll_interval_seconds: int = Field(default=5, ge=1)
    worker_tags: list[str] = Field(default_factory=list)


class BatchExecutionBackendSpec(SpecBase):
    """Async backend shape for externally polled batch systems.

    This spec models persisted batch work and polling cadence, not a built-in
    provider adapter. Submission and import still belong to your engine or
    external worker layer.
    """

    kind: Literal["batch"] = Field(default="batch")
    provider: str = Field(..., min_length=1)
    poll_interval_seconds: int = Field(default=30, ge=1)
    max_batch_items: int = Field(default=250, ge=1)


ExecutionBackendConfig = Annotated[
    LocalExecutionBackendSpec
    | WorkerPoolExecutionBackendSpec
    | BatchExecutionBackendSpec,
    Field(discriminator="kind"),
]


class ItemSamplingSpec(SpecBase):
    """Declarative benchmark slicing and deterministic sampling controls."""

    kind: SamplingKind = Field(default=SamplingKind.ALL)
    count: int | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None)
    strata_field: str | None = Field(
        default=None, description="Field name used for stratified sampling."
    )
    item_ids: list[str] = Field(
        default_factory=list,
        description="Optional allow-list of dataset item IDs to retain before sampling.",
    )
    metadata_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Exact-match metadata filters applied before subset/stratified sampling.",
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: SamplingKind | str) -> SamplingKind | str:
        if isinstance(value, str):
            return SamplingKind(value)
        return value

    @classmethod
    def all(cls) -> ItemSamplingSpec:
        """Return a sampling config that keeps all dataset items."""
        return cls(kind=SamplingKind.ALL)

    @classmethod
    def subset(cls, count: int, seed: int | None = None) -> ItemSamplingSpec:
        """Return a deterministic subset-sampling configuration."""
        return cls(kind=SamplingKind.SUBSET, count=count, seed=seed)

    @classmethod
    def stratified(
        cls,
        count: int,
        *,
        strata_field: str,
        seed: int | None = None,
    ) -> ItemSamplingSpec:
        """Return a stratified-sampling configuration for one dataset field."""
        return cls(
            kind=SamplingKind.STRATIFIED,
            count=count,
            strata_field=strata_field,
            seed=seed,
        )

    @model_validator(mode="after")
    def _validate_semantic(self) -> ItemSamplingSpec:
        if (
            self.kind in {SamplingKind.SUBSET, SamplingKind.STRATIFIED}
            and self.count is None
        ):
            raise ValueError(
                f"ItemSamplingSpec kind='{self.kind.value}' requires a positive count."
            )
        if self.kind == SamplingKind.STRATIFIED and not self.strata_field:
            raise ValueError(
                "ItemSamplingSpec kind='stratified' requires strata_field."
            )
        return self


class InferenceGridSpec(SpecBase):
    """Typed inference sweep over base params and scalar override grids.

    Use this for temperature, top-p, or provider-extra sweeps while keeping
    unchanged parameter combinations resumable across runs.
    """

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
    """Author-facing experiment matrix expanded into `TrialSpec` objects.

    This is the code-first configuration surface for model lists, prompt sweeps,
    inference grids, benchmark slices, and task-local stage composition.
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
    """Shared project-level identity, storage defaults, and execution policy.

    Keep this stable across related experiment runs so resume behavior and run
    manifests refer to the same storage and backend context.
    """

    project_name: str = Field(..., description="Human readable project name.")
    researcher_id: str = Field(
        ..., description="Stable owner identifier for experiment lineage."
    )
    global_seed: int = Field(
        ..., description="Default deterministic seed shared across experiments."
    )
    storage: StorageConfig = Field(..., description="Shared storage defaults.")
    execution_policy: ExecutionPolicySpec = Field(
        ..., description="Shared retry and circuit-breaker policy."
    )
    execution_backend: ExecutionBackendConfig = Field(
        default_factory=LocalExecutionBackendSpec,
        description="Execution backend used for local, worker-pool, or batch orchestration.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="User-defined project metadata."
    )


JudgeInferenceSpec.model_rebuild(
    _types_namespace={"InferenceParamsSpec": InferenceParamsSpec}
)
