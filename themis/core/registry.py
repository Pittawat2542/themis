"""Run registry models and helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from themis.core.base import FrozenModel
from themis.core.results import ExecutionState
from themis.core.snapshot import RunSnapshot


def _now_utc() -> datetime:
    return datetime.now(UTC)


class RunLineage(FrozenModel):
    """Relationship from one run to a parent run."""

    parent_run_id: str
    relationship: str = "derived"


class RunRecord(FrozenModel):
    """Searchable registry entry for one persisted run."""

    run_id: str
    status: str = "pending"
    dataset_source_ids: list[str] = Field(default_factory=list)
    dataset_fingerprints: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    baseline_label: str | None = None
    lineage: list[RunLineage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)


class RunQuery(FrozenModel):
    """Filter set for querying persisted run records."""

    run_id: str | None = None
    dataset_source_id: str | None = None
    dataset_fingerprint: str | None = None
    metric_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    baseline_label: str | None = None
    lineage_parent_run_id: str | None = None
    status: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    updated_after: datetime | None = None
    updated_before: datetime | None = None


class RegressionPolicy(FrozenModel):
    """Threshold policy for comparing one candidate run against a baseline."""

    baseline_label: str
    metric_thresholds: dict[str, float] = Field(default_factory=dict)


def build_run_record(
    snapshot: RunSnapshot,
    *,
    state: ExecutionState | None = None,
    existing: RunRecord | None = None,
) -> RunRecord:
    """Build or refresh a run record from snapshot and execution state."""

    return RunRecord(
        run_id=snapshot.run_id,
        status=existing.status
        if existing is not None and state is None
        else (state.status.value if state is not None else "pending"),
        dataset_source_ids=[
            dataset_ref.dataset_id
            for dataset_ref in snapshot.identity.dataset_source_refs
        ],
        dataset_fingerprints=[
            dataset_ref.fingerprint
            for dataset_ref in snapshot.identity.dataset_source_refs
        ],
        metric_ids=[
            component_ref.component_id
            for component_ref in snapshot.component_refs.metrics
        ],
        tags=[] if existing is None else list(existing.tags),
        baseline_label=None if existing is None else existing.baseline_label,
        lineage=[] if existing is None else list(existing.lineage),
        created_at=_now_utc() if existing is None else existing.created_at,
        updated_at=_now_utc(),
    )


def matches_run_query(record: RunRecord, query: RunQuery) -> bool:
    """Return whether a run record matches the provided query."""

    if query.run_id is not None and record.run_id != query.run_id:
        return False
    if (
        query.dataset_source_id is not None
        and query.dataset_source_id not in record.dataset_source_ids
    ):
        return False
    if (
        query.dataset_fingerprint is not None
        and query.dataset_fingerprint not in record.dataset_fingerprints
    ):
        return False
    if query.metric_id is not None and query.metric_id not in record.metric_ids:
        return False
    if query.tags and any(tag not in record.tags for tag in query.tags):
        return False
    if (
        query.baseline_label is not None
        and record.baseline_label != query.baseline_label
    ):
        return False
    if query.status is not None and record.status != query.status:
        return False
    if query.lineage_parent_run_id is not None and query.lineage_parent_run_id not in {
        lineage.parent_run_id for lineage in record.lineage
    }:
        return False
    if query.created_after is not None and record.created_at < query.created_after:
        return False
    if query.created_before is not None and record.created_at > query.created_before:
        return False
    if query.updated_after is not None and record.updated_at < query.updated_after:
        return False
    if query.updated_before is not None and record.updated_at > query.updated_before:
        return False
    return True
