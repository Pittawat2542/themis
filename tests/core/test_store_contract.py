from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.inspection import get_execution_state
from themis.core.models import Case, Dataset
from themis.core.registry import RunLineage, RunQuery
from themis.core.results import ExecutionCheckpoint, ExecutionState, ProjectionCursor
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.storage import AppendResult, EventRecord, RunStoreBase
from themis.core.stores import (
    InMemoryRunStore,
    jsonl_store,
    mongodb_store,
    postgres_store,
    sqlite_store,
)
from tests.release import CURRENT_VERSION
from tests.core.store_fakes import fake_psycopg_module, fake_pymongo_module


class MinimalRunStore(RunStoreBase):
    """Test backend implementing only the public persistence primitives."""

    storage_config = StorageConfig(target="memory", kwargs={"backend": "minimal"})

    def __init__(self) -> None:
        self.snapshots: dict[str, RunSnapshot] = {}
        self.events: dict[str, list[RunEvent]] = {}
        self.event_sequences: dict[tuple[str, str], int] = {}
        self.documents: dict[tuple[str, str], JSONValue] = {}
        self.blobs: dict[str, tuple[str, bytes]] = {}

    def initialize(self) -> None:
        pass

    def write_snapshot(self, snapshot: RunSnapshot) -> None:
        self.snapshots[snapshot.run_id] = snapshot

    def read_snapshot(self, run_id: str) -> RunSnapshot | None:
        return self.snapshots.get(run_id)

    def append_event(self, event: RunEvent) -> AppendResult:
        key = (event.run_id, event.event_id)
        if key in self.event_sequences:
            return AppendResult(sequence=self.event_sequences[key], inserted=False)
        sequence = len(self.events.setdefault(event.run_id, [])) + 1
        self.events[event.run_id].append(event)
        self.event_sequences[key] = sequence
        return AppendResult(sequence=sequence, inserted=True)

    def read_event_records(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> list[EventRecord]:
        return [
            EventRecord(sequence=sequence, event=event)
            for sequence, event in enumerate(self.events.get(run_id, []), start=1)
            if sequence > after_sequence
        ][:limit]

    def list_run_ids(self) -> list[str]:
        return list(self.snapshots)

    def write_document(self, collection: str, key: str, payload: JSONValue) -> None:
        self.documents[(collection, key)] = payload

    def read_document(self, collection: str, key: str) -> JSONValue | None:
        return self.documents.get((collection, key))

    def delete_document(self, collection: str, key: str) -> None:
        self.documents.pop((collection, key), None)

    def store_blob(self, blob: bytes, media_type: str) -> str:
        ref = f"sha256:{hashlib.sha256(blob).hexdigest()}"
        self.blobs.setdefault(ref, (media_type, blob))
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        return self.blobs.get(blob_ref)

    def delete_run(self, run_id: str) -> None:
        self.snapshots.pop(run_id, None)
        self.events.pop(run_id, None)
        self.event_sequences = {
            key: sequence
            for key, sequence in self.event_sequences.items()
            if key[0] != run_id
        }
        self.documents = {
            key: payload
            for key, payload in self.documents.items()
            if run_id not in key[1]
        }


def _snapshot() -> RunSnapshot:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(target="memory", kwargs={"path": ":memory:"}),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
                revision="r1",
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version=CURRENT_VERSION,
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


def _store(label: str, tmp_path: Path, monkeypatch) -> RunStore:
    if label == "memory":
        return InMemoryRunStore()
    if label == "sqlite":
        return sqlite_store(tmp_path / "run_store.sqlite3")
    if label == "jsonl":
        return jsonl_store(tmp_path / "jsonl-store")
    if label == "mongodb":
        monkeypatch.setattr(
            "themis.core.stores.mongodb.importlib.import_module",
            lambda name: fake_pymongo_module(),
        )
        return mongodb_store(
            "mongodb://example", "themis_test", tmp_path / "mongodb-blobs"
        )
    if label == "postgres":
        monkeypatch.setattr(
            "themis.core.stores.postgres.importlib.import_module",
            lambda name: fake_psycopg_module(),
        )
        return postgres_store(str(tmp_path / "postgres.sqlite3"), tmp_path / "pg-blobs")
    raise AssertionError(label)


def test_run_store_base_derives_the_full_contract_from_primitives() -> None:
    store = MinimalRunStore()
    snapshot = _snapshot()
    started = RunStartedEvent(run_id=snapshot.run_id)
    completed = RunCompletedEvent(run_id=snapshot.run_id)

    store.initialize()
    store.persist_snapshot(snapshot)
    assert store.persist_event(started).inserted is True
    assert store.persist_event(started).inserted is False
    store.persist_event(completed)

    resumed = store.resume(snapshot.run_id)
    assert resumed is not None
    assert resumed.snapshot == snapshot
    assert [event.event_type for event in resumed.events] == [
        "run_started",
        "run_completed",
    ]
    projection = store.get_projection(snapshot.run_id, "run_result")
    assert isinstance(projection, dict)
    assert projection["status"] == "completed"

    store.update_run_record(snapshot.run_id, tags=["custom"])
    assert store.query_runs(RunQuery(tags=["custom"]))[0].run_id == snapshot.run_id

    store.store_stage_cache("generate", "key", {"candidate": "4"})
    assert store.load_stage_cache("generate", "key") == {"candidate": "4"}
    blob_ref = store.store_blob(b"payload", "application/octet-stream")
    assert store.load_blob(blob_ref) == ("application/octet-stream", b"payload")

    store.clear_run(snapshot.run_id)
    assert store.resume(snapshot.run_id) is None


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb", "postgres"],
)
def test_run_store_event_and_projection_lifecycle(
    label: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()
    started = RunStartedEvent(run_id=snapshot.run_id)

    store.initialize()
    store.persist_snapshot(snapshot)
    snapshot_projection = store.get_projection(snapshot.run_id, "snapshot")
    first = store.persist_event(started)
    duplicate = store.persist_event(started)
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    resumed = store.resume(snapshot.run_id)
    events = store.query_events(snapshot.run_id)
    records = store.query_event_records(snapshot.run_id)
    second_page = store.query_event_records(snapshot.run_id, after_sequence=1, limit=1)
    run_result = store.get_projection(snapshot.run_id, "run_result")
    benchmark_result = store.get_projection(snapshot.run_id, "benchmark_result")
    timeline_view = store.get_projection(snapshot.run_id, "timeline_view")
    trace_view = store.get_projection(snapshot.run_id, "trace_view")

    assert snapshot_projection == snapshot.model_dump(mode="json")
    assert first == AppendResult(sequence=1, inserted=True)
    assert duplicate == AppendResult(sequence=1, inserted=False)
    assert store.count_events(snapshot.run_id) == 2
    assert resumed is not None
    assert resumed.snapshot == snapshot
    assert [event.event_type for event in events] == ["run_started", "run_completed"]
    assert [record.sequence for record in records] == [1, 2]
    assert [record.event.event_type for record in records] == [
        "run_started",
        "run_completed",
    ]
    assert [record.sequence for record in second_page] == [2]
    assert isinstance(run_result, dict)
    assert run_result["run_id"] == snapshot.run_id
    assert run_result["status"] == "completed"
    assert isinstance(benchmark_result, dict)
    assert benchmark_result["run_id"] == snapshot.run_id
    assert isinstance(timeline_view, dict)
    entries = cast(list[JSONValue], timeline_view["entries"])
    assert [cast(dict[str, JSONValue], entry)["event_type"] for entry in entries] == [
        "run_started",
        "run_completed",
    ]
    assert isinstance(trace_view, dict)
    assert trace_view["generation_traces"] == []


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb", "postgres"],
)
def test_run_store_blob_lifecycle(label: str, tmp_path: Path, monkeypatch) -> None:
    store = _store(label, tmp_path, monkeypatch)

    store.initialize()
    left = store.store_blob(b'{"answer":"4"}', "application/json")
    right = store.store_blob(b'{"answer":"4"}', "application/json")

    assert left == right
    assert store.load_blob(left) == ("application/json", b'{"answer":"4"}')


def test_in_memory_store_updates_projections_without_resume_replay() -> None:
    class NoReplayInMemoryRunStore(InMemoryRunStore):
        def resume(self, run_id: str):
            raise AssertionError(
                f"resume should not be used while persisting projections for {run_id}"
            )

    store = NoReplayInMemoryRunStore()
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    run_result = store.get_projection(snapshot.run_id, "run_result")

    assert isinstance(run_result, dict)
    assert run_result["status"] == "completed"


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb", "postgres"],
)
def test_run_store_metadata_checkpoint_and_clear_lifecycle(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.update_run_record(
        snapshot.run_id,
        tags=["phase1", "smoke"],
        baseline_label="main",
        lineage=[RunLineage(parent_run_id="parent-run", relationship="rerun")],
    )
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    state = ExecutionState.from_events(
        snapshot.run_id, store.query_events(snapshot.run_id)
    )
    checkpoint = ExecutionCheckpoint(
        run_id=snapshot.run_id,
        event_count=store.count_events(snapshot.run_id),
        execution_state=state,
    )
    cursor = ProjectionCursor(
        run_id=snapshot.run_id,
        projection_name="execution_state",
        event_count=store.count_events(snapshot.run_id),
    )
    store.store_execution_checkpoint(checkpoint)
    store.store_projection_cursor(cursor)

    assert store.load_execution_checkpoint(snapshot.run_id) == checkpoint
    assert store.load_projection_cursor(snapshot.run_id, "execution_state") == cursor

    record = store.get_run_record(snapshot.run_id)
    matches = store.query_runs(
        RunQuery(
            run_id=snapshot.run_id,
            dataset_source_id="dataset-1",
            dataset_fingerprint=snapshot.identity.dataset_source_refs[0].fingerprint,
            metric_id="builtin/exact_match",
            tags=["phase1"],
            baseline_label="main",
            lineage_parent_run_id="parent-run",
        )
    )

    assert record is not None
    assert record.status == "completed"
    assert record.tags == ["phase1", "smoke"]
    assert record.baseline_label == "main"
    assert record.lineage[0].parent_run_id == "parent-run"
    assert record.created_at <= record.updated_at
    assert [item.run_id for item in matches] == [snapshot.run_id]

    store.clear_run(snapshot.run_id)

    assert store.resume(snapshot.run_id) is None
    assert store.load_execution_checkpoint(snapshot.run_id) is None
    assert store.load_projection_cursor(snapshot.run_id, "execution_state") is None


def test_get_execution_state_uses_fresh_checkpoint_without_event_replay() -> None:
    class NoEventReplayStore(InMemoryRunStore):
        def query_events(self, run_id: str):
            raise AssertionError(f"query_events should not be called for {run_id}")

    store = NoEventReplayStore()
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    state = ExecutionState(run_id=snapshot.run_id)
    store.store_execution_checkpoint(
        ExecutionCheckpoint(
            run_id=snapshot.run_id,
            event_count=store.count_events(snapshot.run_id),
            execution_state=state,
        )
    )

    assert get_execution_state(store, snapshot.run_id) == state


def test_get_execution_state_replays_events_when_checkpoint_is_stale() -> None:
    class CountingReplayStore(InMemoryRunStore):
        def __init__(self) -> None:
            super().__init__()
            self.query_calls = 0

        def query_events(self, run_id: str):
            self.query_calls += 1
            return super().query_events(run_id)

    store = CountingReplayStore()
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.store_execution_checkpoint(
        ExecutionCheckpoint(
            run_id=snapshot.run_id,
            event_count=0,
            execution_state=ExecutionState(run_id=snapshot.run_id),
        )
    )
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.store_execution_checkpoint(
        ExecutionCheckpoint(
            run_id=snapshot.run_id,
            event_count=0,
            execution_state=ExecutionState(run_id=snapshot.run_id),
        )
    )

    state = get_execution_state(store, snapshot.run_id)

    assert state.status.value == "running"
    assert store.query_calls >= 1
