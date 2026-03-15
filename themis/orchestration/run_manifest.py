"""Public models for planned runs, stage work items, and external stage bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from themis.records.candidate import CandidateRecord
from themis.specs.experiment import (
    DataItemContext,
    ExperimentSpec,
    ProjectSpec,
    TrialSpec,
)
from themis.types.enums import RunStage


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class WorkItemStatus(str, Enum):
    """Lifecycle state for one persisted stage work item."""

    PENDING = "pending"
    COMPLETED = "completed"


class RunStatus(str, Enum):
    """Lifecycle state for a planned or executing run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


class StageWorkItem(BaseModel):
    """Deterministic execution work item for one stage/candidate combination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work_item_id: str
    stage: RunStage
    status: WorkItemStatus = WorkItemStatus.PENDING
    trial_hash: str
    candidate_index: int
    candidate_id: str
    transform_hash: str | None = None
    evaluation_hash: str | None = None
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    external_job_id: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Canonical snapshot of one planned experiment matrix and its work items."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend_kind: str
    project_spec: ProjectSpec | None = None
    experiment_spec: ExperimentSpec
    trial_hashes: list[str] = Field(default_factory=list)
    transform_hashes: list[str] = Field(default_factory=list)
    evaluation_hashes: list[str] = Field(default_factory=list)
    work_items: list[StageWorkItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now_utc)


class RunDiff(BaseModel):
    """High-level diff between two experiment plans under one project context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_hash_before: str | None = None
    project_hash_after: str | None = None
    experiment_hash_before: str
    experiment_hash_after: str
    changed_project_fields: list[str] = Field(default_factory=list)
    changed_experiment_fields: list[str] = Field(default_factory=list)
    added_trial_hashes: list[str] = Field(default_factory=list)
    removed_trial_hashes: list[str] = Field(default_factory=list)
    added_transform_hashes: list[str] = Field(default_factory=list)
    removed_transform_hashes: list[str] = Field(default_factory=list)
    added_evaluation_hashes: list[str] = Field(default_factory=list)
    removed_evaluation_hashes: list[str] = Field(default_factory=list)


class RunHandle(BaseModel):
    """Operator-facing snapshot of run progress for local or async execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend_kind: str
    status: RunStatus
    total_work_items: int
    pending_work_items: int
    completed_work_items: int
    trial_hashes: list[str] = Field(default_factory=list)
    transform_hashes: list[str] = Field(default_factory=list)
    evaluation_hashes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    """Best-effort preflight estimate for one planned experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend_kind: str
    total_work_items: int
    work_items_by_stage: dict[str, int] = Field(default_factory=dict)
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_total_tokens: int
    estimated_total_cost: float | None = None
    notes: list[str] = Field(default_factory=list)


class GenerationBundleItem(BaseModel):
    """One exported generation work item for an external generation system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work_item_id: str
    trial_hash: str
    candidate_index: int
    candidate_id: str
    trial_spec: TrialSpec
    dataset_context: DataItemContext


class EvaluationBundleItem(BaseModel):
    """One exported evaluation work item for an external scoring or judge system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    work_item_id: str
    trial_hash: str
    candidate_index: int
    candidate_id: str
    transform_hash: str | None = None
    evaluation_hash: str
    trial_spec: TrialSpec
    dataset_context: DataItemContext
    candidate: CandidateRecord


class GenerationWorkBundle(BaseModel):
    """Deterministic generation handoff bundle for external systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Literal[RunStage.GENERATION] = RunStage.GENERATION
    manifest: RunManifest
    items: list[GenerationBundleItem] = Field(default_factory=list)


class EvaluationWorkBundle(BaseModel):
    """Deterministic evaluation handoff bundle for external systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Literal[RunStage.EVALUATION] = RunStage.EVALUATION
    manifest: RunManifest
    items: list[EvaluationBundleItem] = Field(default_factory=list)
