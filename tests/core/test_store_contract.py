from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.inspection import get_execution_state
from themis.core.models import Case, Dataset
from themis.core.registry import RunLineage, RunQuery
from themis.core.results import ExecutionCheckpoint, ExecutionState, ProjectionCursor
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.stores import (
    InMemoryRunStore,
    jsonl_store,
    mongodb_store,
    postgres_store,
    sqlite_store,
)
from tests.release import CURRENT_VERSION
from tests.core.store_fakes import fake_psycopg_module, fake_pymongo_module


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


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_contract_round_trips_snapshot_and_events(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    resumed = store.resume(snapshot.run_id)
    events = store.query_events(snapshot.run_id)

    assert resumed is not None
    assert resumed.snapshot == snapshot
    assert [event.event_type for event in events] == ["run_started", "run_completed"]


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_contract_deduplicates_blob_content(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)

    store.initialize()
    left = store.store_blob(b'{"answer":"4"}', "application/json")
    right = store.store_blob(b'{"answer":"4"}', "application/json")

    assert left == right


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_contract_loads_blob_content(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)

    store.initialize()
    ref = store.store_blob(b'{"answer":"4"}', "application/json")

    assert store.load_blob(ref) == ("application/json", b'{"answer":"4"}')


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_exposes_snapshot_projection(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)

    projection = store.get_projection(snapshot.run_id, "snapshot")

    assert projection == snapshot.model_dump(mode="json")


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_refreshes_read_model_projections_after_event_writes(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    run_result = store.get_projection(snapshot.run_id, "run_result")
    benchmark_result = store.get_projection(snapshot.run_id, "benchmark_result")
    timeline_view = store.get_projection(snapshot.run_id, "timeline_view")
    trace_view = store.get_projection(snapshot.run_id, "trace_view")

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
    ["memory", "sqlite", "jsonl", "mongodb"],
)
def test_run_store_registry_round_trips_queryable_run_metadata(
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


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb", "postgres"],
)
def test_run_store_contract_round_trips_execution_checkpoints_and_projection_cursors(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
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


@pytest.mark.parametrize(
    "label",
    ["memory", "sqlite", "jsonl", "mongodb", "postgres"],
)
def test_run_store_contract_clears_execution_checkpoints_and_projection_cursors(
    label: str, tmp_path: Path, monkeypatch
) -> None:
    store = _store(label, tmp_path, monkeypatch)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.store_execution_checkpoint(
        ExecutionCheckpoint(
            run_id=snapshot.run_id,
            event_count=store.count_events(snapshot.run_id),
            execution_state=ExecutionState.from_events(
                snapshot.run_id, store.query_events(snapshot.run_id)
            ),
        )
    )
    store.store_projection_cursor(
        ProjectionCursor(
            run_id=snapshot.run_id,
            projection_name="execution_state",
            event_count=store.count_events(snapshot.run_id),
        )
    )

    store.clear_run(snapshot.run_id)

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
