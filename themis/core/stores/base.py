"""Shared store helpers for projection-refreshing backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from themis.core.base import JSONValue
from themis.core.events import RunEvent
from themis.core.registry import (
    RunLineage,
    RunQuery,
    RunRecord,
    build_run_record,
    matches_run_query,
)
from themis.core.projections import (
    STORE_PROJECTION_NAMES,
    apply_event_to_store_projection_payloads,
    build_initial_store_projection_payloads,
    build_store_projection_payloads,
)
from themis.core.snapshot import RunSnapshot
from themis.core.store import (
    ProjectionConsistency,
    ProjectionFreshness,
    ProjectionRead,
)
from themis.core.results import ExecutionCheckpoint, ExecutionState, ProjectionCursor


class ProjectionRefreshingStore(ABC):
    """Mixin for stores that maintain projection documents alongside events."""

    @abstractmethod
    def resume(self, run_id: str): ...

    @abstractmethod
    def query_events(self, run_id: str) -> list[RunEvent]: ...

    @abstractmethod
    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None: ...

    @abstractmethod
    def _read_projection(
        self, run_id: str, projection_name: str
    ) -> JSONValue | None: ...

    @abstractmethod
    def _load_snapshot(self, run_id: str) -> RunSnapshot | None: ...

    @abstractmethod
    def _write_run_record(self, run_id: str, record: RunRecord) -> None: ...

    @abstractmethod
    def _read_run_record(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    def _list_run_records(self) -> list[RunRecord]: ...

    @abstractmethod
    def count_events(self, run_id: str) -> int: ...

    @abstractmethod
    def load_execution_checkpoint(self, run_id: str) -> ExecutionCheckpoint | None: ...

    @abstractmethod
    def store_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None: ...

    @abstractmethod
    def load_projection_cursor(
        self, run_id: str, projection_name: str
    ) -> ProjectionCursor | None: ...

    @abstractmethod
    def store_projection_cursor(self, cursor: ProjectionCursor) -> None: ...

    def _bootstrap_projections(self, snapshot: RunSnapshot) -> None:
        event_count = self.count_events(snapshot.run_id)
        events = self.query_events(snapshot.run_id) if event_count else []
        payloads = (
            build_store_projection_payloads(
                snapshot, events
            )
            if event_count
            else build_initial_store_projection_payloads(snapshot)
        )
        execution_state = ExecutionState(run_id=snapshot.run_id)
        for projection_name, payload in payloads.items():
            self._write_projection(snapshot.run_id, projection_name, payload)
            self.store_projection_cursor(
                ProjectionCursor(
                    run_id=snapshot.run_id,
                    projection_name=projection_name,
                    event_count=event_count,
                )
            )
            if projection_name == "execution_state" and isinstance(payload, dict):
                execution_state = ExecutionState.model_validate(payload)
        self.store_execution_checkpoint(
            ExecutionCheckpoint(
                run_id=snapshot.run_id,
                attempt_id=next(
                    (event.attempt_id for event in reversed(events) if event.attempt_id),
                    None,
                ),
                event_count=event_count,
                execution_state=execution_state,
            )
        )
        self._write_run_record(snapshot.run_id, build_run_record(snapshot))

    def _refresh_projections_for_event(
        self, snapshot: RunSnapshot, event: RunEvent
    ) -> None:
        projections = self._store_projections(snapshot.run_id)
        if any(projections.get(name) is None for name in STORE_PROJECTION_NAMES):
            self._backfill_projections(snapshot.run_id)
            projections = self._store_projections(snapshot.run_id)
        event_count = self.count_events(snapshot.run_id)
        for projection_name, payload in apply_event_to_store_projection_payloads(
            snapshot, projections, event
        ).items():
            self._write_projection(snapshot.run_id, projection_name, payload)
            self.store_projection_cursor(
                ProjectionCursor(
                    run_id=snapshot.run_id,
                    projection_name=projection_name,
                    event_count=event_count,
                )
            )
            if projection_name == "execution_state" and isinstance(payload, dict):
                self.store_execution_checkpoint(
                    ExecutionCheckpoint(
                        run_id=snapshot.run_id,
                        attempt_id=event.attempt_id or None,
                        event_count=event_count,
                        execution_state=ExecutionState.model_validate(payload),
                    )
                )
        self._refresh_run_record(snapshot)

    def _get_projection_with_backfill(
        self, run_id: str, projection_name: str
    ) -> JSONValue | None:
        return self.read_projection(run_id, projection_name).payload

    def read_projection(
        self,
        run_id: str,
        projection_name: str,
        *,
        consistency: ProjectionConsistency = ProjectionConsistency.FRESH,
    ) -> ProjectionRead:
        payload = self._read_projection(run_id, projection_name)
        cursor = self.load_projection_cursor(run_id, projection_name)
        event_count = self.count_events(run_id)
        freshness = (
            ProjectionFreshness.MISSING
            if payload is None
            else ProjectionFreshness.FRESH
            if cursor is not None and cursor.event_count == event_count
            else ProjectionFreshness.STALE
        )
        if (
            consistency is ProjectionConsistency.FRESH
            and freshness is not ProjectionFreshness.FRESH
        ):
            self._backfill_projections(run_id, requested_projection=projection_name)
            payload = self._read_projection(run_id, projection_name)
            freshness = (
                ProjectionFreshness.FRESH
                if payload is not None
                else ProjectionFreshness.MISSING
            )
        return ProjectionRead(payload=payload, freshness=freshness)

    def _backfill_projections(
        self, run_id: str, *, requested_projection: str | None = None
    ) -> None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return
        events = self.query_events(run_id)
        event_count = self.count_events(run_id)
        for projection_name, payload in build_store_projection_payloads(
            snapshot, events
        ).items():
            if (
                requested_projection is not None
                and projection_name != requested_projection
            ):
                continue
            self._write_projection(run_id, projection_name, payload)
            self.store_projection_cursor(
                ProjectionCursor(
                    run_id=run_id,
                    projection_name=projection_name,
                    event_count=event_count,
                )
            )
            if projection_name == "execution_state" and isinstance(payload, dict):
                self.store_execution_checkpoint(
                    ExecutionCheckpoint(
                        run_id=run_id,
                        attempt_id=next(
                            (
                                event.attempt_id
                                for event in reversed(events)
                                if event.attempt_id
                            ),
                            None,
                        ),
                        event_count=event_count,
                        execution_state=ExecutionState.model_validate(payload),
                    )
                )

    def _store_projections(self, run_id: str) -> dict[str, JSONValue | None]:
        return {
            projection_name: self._read_projection(run_id, projection_name)
            for projection_name in STORE_PROJECTION_NAMES
        }

    def get_run_record(self, run_id: str) -> RunRecord | None:
        self._backfill_projections(run_id)
        snapshot = self._load_snapshot(run_id)
        if snapshot is not None:
            self._refresh_run_record(snapshot)
        return self._read_run_record(run_id)

    def query_runs(self, query: RunQuery | None = None) -> list[RunRecord]:
        for record in self._list_run_records():
            self.get_run_record(record.run_id)
        records = self._list_run_records()
        if query is None:
            return sorted(records, key=lambda item: item.created_at)
        return [
            record
            for record in sorted(records, key=lambda item: item.created_at)
            if matches_run_query(record, query)
        ]

    def update_run_record(
        self,
        run_id: str,
        *,
        tags: list[str] | None = None,
        baseline_label: str | None = None,
        lineage: list[RunLineage] | None = None,
    ) -> None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            raise ValueError(f"Unknown run_id: {run_id}")
        current = self._read_run_record(run_id)
        if current is None:
            current = build_run_record(snapshot)
        if lineage is not None and any(
            item.parent_run_id == run_id and item.parent_attempt_id is None
            for item in lineage
        ):
            raise ValueError(
                "Same-run lineage must identify a parent_attempt_id"
            )
        self._write_run_record(
            run_id,
            current.model_copy(
                update={
                    "tags": list(tags) if tags is not None else list(current.tags),
                    "baseline_label": (
                        baseline_label
                        if baseline_label is not None
                        else current.baseline_label
                    ),
                    "lineage": list(lineage)
                    if lineage is not None
                    else list(current.lineage),
                }
            ),
        )

    def _refresh_run_record(self, snapshot: RunSnapshot) -> None:
        existing = self._read_run_record(snapshot.run_id)
        state_payload = self._read_projection(snapshot.run_id, "execution_state")
        state = (
            ExecutionState.model_validate(state_payload)
            if isinstance(state_payload, dict)
            else None
        )
        self._write_run_record(
            snapshot.run_id,
            build_run_record(snapshot, state=state, existing=existing),
        )
