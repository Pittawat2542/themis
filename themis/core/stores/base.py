"""Shared store helpers for projection-refreshing backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from themis.core.base import JSONValue
from themis.core.projections import build_projection_payloads


class ProjectionRefreshingStore(ABC):
    @abstractmethod
    def resume(self, run_id: str): ...

    @abstractmethod
    def _write_projection(self, run_id: str, projection_name: str, payload: JSONValue) -> None: ...

    def _refresh_projections(self, run_id: str) -> None:
        stored = self.resume(run_id)
        if stored is None:
            return
        for projection_name, payload in build_projection_payloads(stored.snapshot, stored.events).items():
            self._write_projection(run_id, projection_name, payload)
