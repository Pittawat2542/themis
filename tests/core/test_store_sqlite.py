from __future__ import annotations

import json
import sqlite3

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
        storage=StorageConfig(store="sqlite", parameters={"path": "runs/themis.sqlite3"}),
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
