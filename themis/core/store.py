"""Run store protocol for Themis."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.registry import RunLineage, RunQuery, RunRecord
from themis.core.results import ExecutionCheckpoint, ProjectionCursor
from themis.core.snapshot import RunSnapshot, StoredRun
from themis.core.base import FrozenModel


class AppendResult(FrozenModel):
    sequence: int
    inserted: bool


class EventRecord(FrozenModel):
    sequence: int
    event: RunEvent


class ProjectionConsistency(StrEnum):
    FRESH = "fresh"
    EVENTUAL = "eventual"


class ProjectionFreshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


class ProjectionRead(FrozenModel):
    payload: JSONValue | None = None
    freshness: ProjectionFreshness


@runtime_checkable
class RunStore(Protocol):
    """Persistence contract used by Themis runtime components."""

    def initialize(self) -> None: ...

    def persist_snapshot(self, snapshot: RunSnapshot) -> None: ...

    def persist_event(self, event: RunEvent) -> AppendResult: ...

    def query_events(self, run_id: str) -> list[RunEvent]: ...

    def query_event_records(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[EventRecord]: ...

    def count_events(self, run_id: str) -> int: ...

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None: ...

    def read_projection(
        self,
        run_id: str,
        projection_name: str,
        *,
        consistency: ProjectionConsistency = ProjectionConsistency.FRESH,
    ) -> ProjectionRead: ...

    def load_execution_checkpoint(self, run_id: str) -> ExecutionCheckpoint | None: ...

    def store_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None: ...

    def load_projection_cursor(
        self, run_id: str, projection_name: str
    ) -> ProjectionCursor | None: ...

    def store_projection_cursor(self, cursor: ProjectionCursor) -> None: ...

    def store_blob(self, blob: bytes, media_type: str) -> str: ...

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None: ...

    def resume(self, run_id: str) -> StoredRun | None: ...

    def get_run_record(self, run_id: str) -> RunRecord | None: ...

    def query_runs(self, query: RunQuery | None = None) -> list[RunRecord]: ...

    def update_run_record(
        self,
        run_id: str,
        *,
        tags: list[str] | None = None,
        baseline_label: str | None = None,
        lineage: list[RunLineage] | None = None,
    ) -> None: ...

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None: ...

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None: ...

    def clear_run(self, run_id: str) -> None: ...
