"""Shared implementation for custom and built-in run stores."""

from __future__ import annotations

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
from themis.core.snapshot import RunSnapshot, StoredRun
from themis.core.store import (
    AppendResult,
    EventRecord,
    ProjectionConsistency,
    ProjectionFreshness,
    ProjectionRead,
)
from themis.core.results import ExecutionCheckpoint, ExecutionState, ProjectionCursor


class RunStoreBase:
    """Derive the full :class:`RunStore` contract from persistence primitives.

    Backend authors implement ``initialize``, snapshot read/write, idempotent event
    append and sequenced reads, run-id listing, generic document operations, blob
    operations, and run deletion. Built-in stores may override derived methods for
    efficiency.
    """

    # Persistence primitives -------------------------------------------------
    # These deliberately raise instead of being abstract methods so existing
    # optimized backends can override the consumer-facing operations directly.

    def initialize(self) -> None:
        raise NotImplementedError

    def write_snapshot(self, snapshot: RunSnapshot) -> None:
        raise NotImplementedError

    def read_snapshot(self, run_id: str) -> RunSnapshot | None:
        raise NotImplementedError

    def append_event(self, event: RunEvent) -> AppendResult:
        raise NotImplementedError

    def read_event_records(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[EventRecord]:
        raise NotImplementedError

    def list_run_ids(self) -> list[str]:
        raise NotImplementedError

    def write_document(self, collection: str, key: str, payload: JSONValue) -> None:
        raise NotImplementedError

    def read_document(self, collection: str, key: str) -> JSONValue | None:
        raise NotImplementedError

    def delete_document(self, collection: str, key: str) -> None:
        raise NotImplementedError

    def store_blob(self, blob: bytes, media_type: str) -> str:
        raise NotImplementedError

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        raise NotImplementedError

    def delete_run(self, run_id: str) -> None:
        raise NotImplementedError

    # Consumer-facing RunStore operations -----------------------------------

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        self.write_snapshot(snapshot)
        self._bootstrap_projections(snapshot)

    def persist_event(self, event: RunEvent) -> AppendResult:
        return self.append_event(event)

    def query_event_records(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[EventRecord]:
        return self.read_event_records(
            run_id, after_sequence=after_sequence, limit=limit
        )

    def query_events(self, run_id: str) -> list[RunEvent]:
        records: list[EventRecord] = []
        after_sequence = 0
        while True:
            page = self.read_event_records(
                run_id, after_sequence=after_sequence, limit=1000
            )
            if not page:
                break
            records.extend(page)
            after_sequence = page[-1].sequence
        return [record.event for record in records]

    def count_events(self, run_id: str) -> int:
        return len(self.query_events(run_id))

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._load_snapshot(run_id)
        if snapshot is None:
            return None
        events = self.query_events(run_id)
        return StoredRun(
            snapshot=snapshot,
            events=events,
            execution_checkpoint=self.load_execution_checkpoint(run_id),
            event_count=len(events),
        )

    def _write_projection(
        self, run_id: str, projection_name: str, payload: JSONValue
    ) -> None:
        self.write_document("projection", f"{run_id}:{projection_name}", payload)

    def _read_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self.read_document("projection", f"{run_id}:{projection_name}")

    def _load_snapshot(self, run_id: str) -> RunSnapshot | None:
        return self.read_snapshot(run_id)

    def _write_run_record(self, run_id: str, record: RunRecord) -> None:
        self.write_document("run_record", run_id, record.model_dump(mode="json"))

    def _read_run_record(self, run_id: str) -> RunRecord | None:
        payload = self.read_document("run_record", run_id)
        return RunRecord.model_validate(payload) if payload is not None else None

    def _list_run_records(self) -> list[RunRecord]:
        return [
            record
            for run_id in self.list_run_ids()
            if (record := self._read_run_record(run_id)) is not None
        ]

    def load_execution_checkpoint(self, run_id: str) -> ExecutionCheckpoint | None:
        payload = self.read_document("execution_checkpoint", run_id)
        return (
            ExecutionCheckpoint.model_validate(payload) if payload is not None else None
        )

    def store_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        self.write_document(
            "execution_checkpoint",
            checkpoint.run_id,
            checkpoint.model_dump(mode="json"),
        )

    def load_projection_cursor(
        self, run_id: str, projection_name: str
    ) -> ProjectionCursor | None:
        payload = self.read_document("projection_cursor", f"{run_id}:{projection_name}")
        return ProjectionCursor.model_validate(payload) if payload is not None else None

    def store_projection_cursor(self, cursor: ProjectionCursor) -> None:
        self.write_document(
            "projection_cursor",
            f"{cursor.run_id}:{cursor.projection_name}",
            cursor.model_dump(mode="json"),
        )

    def get_projection(self, run_id: str, projection_name: str) -> JSONValue | None:
        return self._get_projection_with_backfill(run_id, projection_name)

    def load_stage_cache(self, stage_name: str, cache_key: str) -> JSONValue | None:
        return self.read_document("stage_cache", f"{stage_name}:{cache_key}")

    def store_stage_cache(
        self, stage_name: str, cache_key: str, payload: JSONValue
    ) -> None:
        self.write_document("stage_cache", f"{stage_name}:{cache_key}", payload)

    def clear_run(self, run_id: str) -> None:
        self.delete_run(run_id)

    def _bootstrap_projections(self, snapshot: RunSnapshot) -> None:
        event_count = self.count_events(snapshot.run_id)
        events = self.query_events(snapshot.run_id) if event_count else []
        payloads = (
            build_store_projection_payloads(snapshot, events)
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
                    (
                        event.attempt_id
                        for event in reversed(events)
                        if event.attempt_id
                    ),
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
            raise ValueError("Same-run lineage must identify a parent_attempt_id")
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


# Internal compatibility name for existing backend modules.
ProjectionRefreshingStore = RunStoreBase
