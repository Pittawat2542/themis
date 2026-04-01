"""Run snapshot models for the Themis v4 runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, computed_field

from themis.core.base import FrozenModel, HashableModel, JSONValue
from themis.core.components import BUILTIN_COMPONENT_REFS, ComponentRef
from themis.core.config import RuntimeConfig, StorageConfig
from themis.core.events import RunEvent
from themis.core.models import Dataset
from themis.core.security import (
    sanitize_persisted_json_value,
    sanitize_persisted_string_mapping,
)

if TYPE_CHECKING:
    from themis.core.results import ExecutionState

__all__ = [
    "BUILTIN_COMPONENT_REFS",
    "ComponentRef",
    "ComponentRefs",
    "DatasetRef",
    "RunIdentity",
    "RunProvenance",
    "RunSnapshot",
    "StoredRun",
    "snapshot_from_dict",
]


class DatasetRef(HashableModel):
    """Identity-bearing reference to one dataset."""

    dataset_id: str
    revision: str | None = None
    fingerprint: str


class ComponentRefs(FrozenModel):
    """Resolved component refs stored with the snapshot."""

    generator: ComponentRef
    selector: ComponentRef | None = None
    reducer: ComponentRef | None = None
    parsers: list[ComponentRef] = Field(default_factory=list)
    metrics: list[ComponentRef] = Field(default_factory=list)
    judge_models: list[ComponentRef] = Field(default_factory=list)


class RunIdentity(HashableModel):
    """Inputs that determine the logical identity and `run_id` of a run."""

    dataset_refs: list[DatasetRef] = Field(default_factory=list)
    generator_ref: ComponentRef
    selector_ref: ComponentRef | None = None
    reducer_ref: ComponentRef | None = None
    parser_refs: list[ComponentRef] = Field(default_factory=list)
    metric_refs: list[ComponentRef] = Field(default_factory=list)
    judge_model_refs: list[ComponentRef] = Field(default_factory=list)
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)

    def sanitized(self) -> RunIdentity:
        return self.model_copy(
            update={
                "candidate_policy": sanitize_persisted_json_value(
                    self.candidate_policy,
                    field_path="identity.candidate_policy",
                ),
                "judge_config": sanitize_persisted_json_value(
                    self.judge_config,
                    field_path="identity.judge_config",
                ),
                "workflow_overrides": sanitize_persisted_json_value(
                    self.workflow_overrides,
                    field_path="identity.workflow_overrides",
                ),
            }
        )


class RunProvenance(FrozenModel):
    """Environment metadata recorded with a run but excluded from `run_id`."""

    themis_version: str
    python_version: str
    platform: str
    git_commit: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    provider_metadata: dict[str, JSONValue] = Field(default_factory=dict)
    storage: StorageConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    environment_metadata: dict[str, str] = Field(default_factory=dict)

    def sanitized(self) -> RunProvenance:
        storage_parameters = sanitize_persisted_json_value(
            self.storage.parameters,
            field_path="provenance.storage.parameters",
        )
        environment_metadata = sanitize_persisted_string_mapping(
            self.environment_metadata,
            field_path="provenance.environment_metadata",
        )
        return self.model_copy(
            update={
                "dependency_versions": dict(self.dependency_versions),
                "provider_metadata": sanitize_persisted_json_value(
                    self.provider_metadata,
                    field_path="provenance.provider_metadata",
                ),
                "storage": self.storage.model_copy(
                    update={"parameters": storage_parameters}
                ),
                "environment_metadata": environment_metadata,
            }
        )


class RunSnapshot(FrozenModel):
    """Immutable executable artifact produced by `Experiment.compile()`."""

    identity: RunIdentity
    provenance: RunProvenance
    component_refs: ComponentRefs
    datasets: list[Dataset] = Field(default_factory=list)
    metric_kinds: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_id(self) -> str:
        return self.identity.compute_hash()


class StoredRun(FrozenModel):
    """Snapshot plus stored events loaded back from a run store."""

    snapshot: RunSnapshot
    events: list[RunEvent] = Field(default_factory=list)

    @computed_field(return_type=object)  # type: ignore[prop-decorator]
    @property
    def execution_state(self) -> ExecutionState:
        from themis.core.results import ExecutionState

        return ExecutionState.from_events(self.snapshot.run_id, self.events)


def snapshot_from_dict(payload: dict[str, Any]) -> RunSnapshot:
    """Load a stored snapshot payload and ignore any cached `run_id` field."""

    normalized = dict(payload)
    normalized.pop("run_id", None)
    return RunSnapshot.model_validate(normalized)
