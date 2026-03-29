"""Run snapshot models for Themis v4 Phase 1."""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field

from themis.core.base import FrozenModel, HashableModel, JSONValue
from themis.core.components import (
    BUILTIN_COMPONENT_REFS,
    ComponentRef,
    component_ref_from_value,
)
from themis.core.config import StorageConfig
from themis.core.events import RunEvent


class DatasetRef(HashableModel):
    dataset_id: str
    revision: str | None = None
    fingerprint: str


class ComponentRefs(FrozenModel):
    generator: ComponentRef
    reducer: ComponentRef | None = None
    parsers: list[ComponentRef] = Field(default_factory=list)
    metrics: list[ComponentRef] = Field(default_factory=list)


class RunIdentity(HashableModel):
    dataset_refs: list[DatasetRef] = Field(default_factory=list)
    generator_ref: ComponentRef
    reducer_ref: ComponentRef | None = None
    parser_refs: list[ComponentRef] = Field(default_factory=list)
    metric_refs: list[ComponentRef] = Field(default_factory=list)
    candidate_policy: dict[str, JSONValue] = Field(default_factory=dict)
    judge_config: dict[str, JSONValue] = Field(default_factory=dict)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)


class RunProvenance(FrozenModel):
    themis_version: str
    python_version: str
    platform: str
    storage: StorageConfig
    environment_metadata: dict[str, str] = Field(default_factory=dict)


class RunSnapshot(FrozenModel):
    identity: RunIdentity
    provenance: RunProvenance
    component_refs: ComponentRefs

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_id(self) -> str:
        return self.identity.compute_hash()


class StoredRun(FrozenModel):
    snapshot: RunSnapshot
    events: list[RunEvent] = Field(default_factory=list)


def snapshot_from_dict(payload: dict[str, Any]) -> RunSnapshot:
    normalized = dict(payload)
    normalized.pop("run_id", None)
    return RunSnapshot.model_validate(normalized)
