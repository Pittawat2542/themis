"""In-memory run store implementation."""

from __future__ import annotations

import hashlib

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.snapshot import RunSnapshot, StoredRun
from themis.core.stores.base import ProjectionRefreshingStore


class InMemoryRunStore(ProjectionRefreshingStore):
    """Simple in-memory store used by tests and local development."""

    def __init__(self) -> None:
        self._snapshots: dict[str, RunSnapshot] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._blobs: dict[str, tuple[str, bytes]] = {}
        self._projections: dict[tuple[str, str], JSONValue] = {}

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

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

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
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._projections.get((run_id, projection_name))

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        return self._snapshots.get(run_id)

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        self._projections[(run_id, projection_name)] = payload
