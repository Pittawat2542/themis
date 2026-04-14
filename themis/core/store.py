"""Run store protocol for Themis."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.registry import RunLineage, RunQuery, RunRecord
from themis.core.snapshot import RunSnapshot, StoredRun


@runtime_checkable
class RunStore(Protocol):
    """Persistence contract used by Themis runtime components."""

    def initialize(self) -> None: ...

    def persist_snapshot(self, snapshot: RunSnapshot) -> None: ...

    def persist_event(self, event: RunEvent) -> None: ...

    def query_events(self, run_id: str) -> list[RunEvent]: ...

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None: ...

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
