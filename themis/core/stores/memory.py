"""In-memory run store implementation."""

from __future__ import annotations

import hashlib

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.registry import RunRecord
from themis.core.results import ExecutionCheckpoint, ProjectionCursor
from themis.core.snapshot import RunSnapshot, StoredRun
from themis.core.stores.base import ProjectionRefreshingStore


class InMemoryRunStore(ProjectionRefreshingStore):
    """Simple in-memory store used by tests and local development."""

    def __init__(self) -> None:
        self._snapshots: dict[str, RunSnapshot] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._blobs: dict[str, tuple[str, bytes]] = {}
        self._projections: dict[tuple[str, str], JSONValue] = {}
        self._run_records: dict[str, RunRecord] = {}
        self._execution_checkpoints: dict[str, ExecutionCheckpoint] = {}
        self._projection_cursors: dict[tuple[str, str], ProjectionCursor] = {}

    def initialize(self) -> None:
        return None

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        self._snapshots[snapshot.run_id] = snapshot
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> None:
        self._events.setdefault(event.run_id, []).append(event)
        snapshot = self._load_snapshot(event.run_id)
        if snapshot is not None:
            self._refresh_projections_for_event(snapshot, event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        return list(self._events.get(run_id, []))

    def count_events(self, run_id: str) -> int:
        return len(self._events.get(run_id, []))

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def load_execution_checkpoint(self, run_id: str) -> ExecutionCheckpoint | None:
        return self._execution_checkpoints.get(run_id)

    def store_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        self._execution_checkpoints[checkpoint.run_id] = checkpoint

    def load_projection_cursor(
        self, run_id: str, projection_name: str
    ) -> ProjectionCursor | None:
        return self._projection_cursors.get((run_id, projection_name))

    def store_projection_cursor(self, cursor: ProjectionCursor) -> None:
        self._projection_cursors[(cursor.run_id, cursor.projection_name)] = cursor

    def store_blob(self, blob: bytes, media_type: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        ref = f"sha256:{digest}"
        self._blobs.setdefault(ref, (media_type, blob))
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        return self._blobs.get(blob_ref)

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._snapshots.get(run_id)
        if snapshot is None:
            return None
        return StoredRun(
            snapshot=snapshot,
            events=self.query_events(run_id),
            execution_checkpoint=self.load_execution_checkpoint(run_id),
            event_count=self.count_events(run_id),
        )

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._projections.get((run_id, projection_name))

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        return self._snapshots.get(run_id)

    def _write_run_record(self, run_id: str, record: RunRecord) -> None:
        self._run_records[run_id] = record

    def _read_run_record(self, run_id: str) -> RunRecord | None:
        return self._run_records.get(run_id)

    def _list_run_records(self) -> list[RunRecord]:
        return list(self._run_records.values())

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        self._projections[(run_id, projection_name)] = payload

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        del stage_name, cache_key
        return None

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        del stage_name, cache_key, payload

    def clear_run(self, run_id: str) -> None:
        self._snapshots.pop(run_id, None)
        self._events.pop(run_id, None)
        self._run_records.pop(run_id, None)
        self._execution_checkpoints.pop(run_id, None)
        stale_projection_keys = [
            projection_key
            for projection_key in self._projections
            if projection_key[0] == run_id
        ]
        for projection_key in stale_projection_keys:
            self._projections.pop(projection_key, None)
        stale_cursor_keys = [
            cursor_key for cursor_key in self._projection_cursors if cursor_key[0] == run_id
        ]
        for cursor_key in stale_cursor_keys:
            self._projection_cursors.pop(cursor_key, None)
