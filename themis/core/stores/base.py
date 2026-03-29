"""Shared store helpers for projection-refreshing backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.projections import (
    STORE_PROJECTION_NAMES,
    apply_event_to_store_projection_payloads,
    build_initial_store_projection_payloads,
    build_store_projection_payloads,
)
from themis.core.snapshot import RunSnapshot


class ProjectionRefreshingStore(ABC):
    @abstractmethod
    def resume(self, run_id: str): ...

    @abstractmethod
    def _write_projection(self, run_id: str, projection_name: str, payload: JSONValue) -> None: ...

    @abstractmethod
    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None: ...

    @abstractmethod
    def _load_snapshot(self, run_id: str) -> RunSnapshot | None: ...

    def _bootstrap_projections(self, snapshot: RunSnapshot) -> None:
        for projection_name, payload in build_initial_store_projection_payloads(snapshot).items():
            self._write_projection(snapshot.run_id, projection_name, payload)

    def _refresh_projections_for_event(self, snapshot: RunSnapshot, event: RunEvent) -> None:
        projections = self._store_projections(snapshot.run_id)
        if any(projections.get(name) is None for name in STORE_PROJECTION_NAMES):
            self._backfill_projections(snapshot.run_id)
            projections = self._store_projections(snapshot.run_id)
        for projection_name, payload in apply_event_to_store_projection_payloads(snapshot, projections, event).items():
            self._write_projection(snapshot.run_id, projection_name, payload)

    def _get_projection_with_backfill(self, run_id: str, projection_name: str) -> JSONValue | None:
        projection = self._read_projection(run_id, projection_name)
        if projection is not None:
            return projection
        self._backfill_projections(run_id)
        return self._read_projection(run_id, projection_name)

    def _backfill_projections(self, run_id: str) -> None:
        stored = self.resume(run_id)
        if stored is None:
            return
        for projection_name, payload in build_store_projection_payloads(stored.snapshot, stored.events).items():
            self._write_projection(run_id, projection_name, payload)

    def _store_projections(self, run_id: str) -> dict[str, JSONValue | None]:
        return {
            projection_name: self._read_projection(run_id, projection_name)
            for projection_name in STORE_PROJECTION_NAMES
        }
