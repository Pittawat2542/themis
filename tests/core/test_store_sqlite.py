from __future__ import annotations

import json
import sqlite3

import themis.core.stores.sqlite as sqlite_module
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.sqlite import SqliteRunStore


def _snapshot():
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
        storage=StorageConfig(
            store="sqlite", parameters={"path": "runs/themis.sqlite3"}
        ),
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
        themis_version="4.0.0",
        python_version="3.12.9",
        platform="macos",
    )
    return experiment.compile()


def test_sqlite_store_skips_unknown_event_types_on_read(tmp_path) -> None:
    path = tmp_path / "run_store.sqlite3"
    store = SqliteRunStore(path)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO run_events (run_id, event_type, event_json)
            VALUES (?, ?, ?)
            """,
            (
                snapshot.run_id,
                "future_event",
                json.dumps(
                    {
                        "schema_version": "2",
                        "event_type": "future_event",
                        "run_id": snapshot.run_id,
                    }
                ),
            ),
        )
        connection.commit()

    events = store.query_events(snapshot.run_id)

    assert [event.event_type for event in events] == ["run_started"]


def test_sqlite_store_backfills_missing_projection_from_snapshot_and_events(
    tmp_path,
) -> None:
    path = tmp_path / "run_store.sqlite3"
    store = SqliteRunStore(path)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            DELETE FROM run_projections
            WHERE run_id = ? AND projection_name = ?
            """,
            (snapshot.run_id, "run_result"),
        )
        connection.commit()

    projection = store.get_projection(snapshot.run_id, "run_result")

    assert isinstance(projection, dict)
    assert projection["run_id"] == snapshot.run_id
    assert projection["status"] == "running"

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT projection_json
            FROM run_projections
            WHERE run_id = ? AND projection_name = ?
            """,
            (snapshot.run_id, "run_result"),
        ).fetchone()

    assert row is not None


def test_sqlite_store_closes_connections_for_stage_cache_operations(
    tmp_path, monkeypatch
) -> None:
    class FakeConnection:
        def __init__(self, *, row: tuple[str] | None = None) -> None:
            self.row = row
            self.closed = False
            self.executed: list[tuple[str, tuple[object, ...]]] = []

        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def execute(
            self, query: str, params: tuple[object, ...] = ()
        ) -> FakeConnection:
            self.executed.append((query, params))
            return self

        def fetchone(self) -> tuple[str] | None:
            return self.row

        def commit(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    queued_connections = [
        FakeConnection(row=('{"ok": true}',)),
        FakeConnection(),
        FakeConnection(),
    ]
    all_connections = list(queued_connections)

    def fake_connect(path: object) -> FakeConnection:
        del path
        return queued_connections.pop(0)

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", fake_connect)
    store = SqliteRunStore(tmp_path / "run_store.sqlite3")

    assert store.load_stage_cache("score", "cache-key") == {"ok": True}
    store.store_stage_cache("score", "cache-key", {"ok": True})
    store.clear_run("run-1")

    assert all(connection.closed for connection in all_connections)
