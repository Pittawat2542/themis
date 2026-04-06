from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.snapshot import RunSnapshot
from themis.core.store import RunStore
from themis.core.stores import (
    InMemoryRunStore,
    jsonl_store,
    mongodb_store,
    sqlite_store,
)
from tests.release import CURRENT_VERSION
from tests.core.store_fakes import fake_pymongo_module


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
        storage=StorageConfig(store="memory", parameters={"path": ":memory:"}),
        datasets=[
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
