from __future__ import annotations

from pathlib import Path

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.stores.memory import InMemoryRunStore
from themis.core.stores.sqlite import SqliteRunStore


def _snapshot() -> RunSnapshot:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(store="memory", parameters={"path": ":memory:"}),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
                revision="r1",
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda tmp_path: InMemoryRunStore()),
        ("sqlite", lambda tmp_path: SqliteRunStore(tmp_path / "run_store.sqlite3")),
    ],
)
def test_run_store_contract_round_trips_snapshot_and_events(label: str, factory, tmp_path: Path) -> None:
    del label
    store: RunStore = factory(tmp_path)
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
    ("label", "factory"),
    [
        ("memory", lambda tmp_path: InMemoryRunStore()),
        ("sqlite", lambda tmp_path: SqliteRunStore(tmp_path / "run_store.sqlite3")),
    ],
)
def test_run_store_contract_deduplicates_blob_content(label: str, factory, tmp_path: Path) -> None:
    del label
    store: RunStore = factory(tmp_path)

    store.initialize()
    left = store.store_blob(b'{"answer":"4"}', "application/json")
    right = store.store_blob(b'{"answer":"4"}', "application/json")

    assert left == right


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda tmp_path: InMemoryRunStore()),
        ("sqlite", lambda tmp_path: SqliteRunStore(tmp_path / "run_store.sqlite3")),
    ],
)
def test_run_store_contract_loads_blob_content(label: str, factory, tmp_path: Path) -> None:
    del label
    store: RunStore = factory(tmp_path)

    store.initialize()
    ref = store.store_blob(b'{"answer":"4"}', "application/json")

    assert store.load_blob(ref) == ("application/json", b'{"answer":"4"}')


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda tmp_path: InMemoryRunStore()),
        ("sqlite", lambda tmp_path: SqliteRunStore(tmp_path / "run_store.sqlite3")),
    ],
)
def test_run_store_exposes_snapshot_projection(label: str, factory, tmp_path: Path) -> None:
    del label
    store: RunStore = factory(tmp_path)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)

    projection = store.get_projection(snapshot.run_id, "snapshot")

    assert projection == snapshot.model_dump(mode="json")


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda tmp_path: InMemoryRunStore()),
        ("sqlite", lambda tmp_path: SqliteRunStore(tmp_path / "run_store.sqlite3")),
    ],
)
def test_run_store_refreshes_read_model_projections_after_event_writes(label: str, factory, tmp_path: Path) -> None:
    del label
    store: RunStore = factory(tmp_path)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    run_result = store.get_projection(snapshot.run_id, "run_result")
    benchmark_result = store.get_projection(snapshot.run_id, "benchmark_result")
    timeline_view = store.get_projection(snapshot.run_id, "timeline_view")
    trace_view = store.get_projection(snapshot.run_id, "trace_view")

    assert run_result is not None
    assert run_result["run_id"] == snapshot.run_id
    assert run_result["status"] == "completed"
    assert benchmark_result is not None
    assert benchmark_result["run_id"] == snapshot.run_id
    assert timeline_view is not None
    assert [entry["event_type"] for entry in timeline_view["entries"]] == ["run_started", "run_completed"]
    assert trace_view is not None
    assert trace_view["generation_traces"] == []
