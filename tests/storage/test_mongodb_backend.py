from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.mongodb import MongoDbRunStore, mongodb_store

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        not os.getenv("THEMIS_TEST_MONGODB_URL"),
        reason="THEMIS_TEST_MONGODB_URL is required for MongoDB integration tests",
    ),
]


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(target="mongodb"),
        dataset_sources=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output="4",
                    )
                ],
            )
        ],
        seeds=[7],
    )
    return experiment.compile()


@pytest.fixture
def mongodb_store_backend(tmp_path: Path):
    database = f"themis_test_{uuid.uuid4().hex}"
    store = mongodb_store(
        os.environ["THEMIS_TEST_MONGODB_URL"],
        database,
        tmp_path / "mongodb-blobs",
    )
    assert isinstance(store, MongoDbRunStore)
    store.initialize()

    try:
        yield store
    finally:
        store._db().client.drop_database(database)


def test_mongodb_store_round_trips_paged_events_and_blobs(
    mongodb_store_backend: MongoDbRunStore,
) -> None:
    store = mongodb_store_backend
    snapshot = _snapshot()
    started = RunStartedEvent(run_id=snapshot.run_id)

    store.persist_snapshot(snapshot)
    first = store.persist_event(started)
    duplicate = store.persist_event(started)
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    blob_ref = store.store_blob(b'{"answer":"4"}', "application/json")

    records = store.query_event_records(snapshot.run_id)
    second_page = store.query_event_records(snapshot.run_id, after_sequence=1, limit=1)

    assert first.sequence == 1
    assert duplicate.inserted is False
    assert duplicate.sequence == first.sequence
    assert [record.sequence for record in records] == [1, 2]
    assert [record.event.event_type for record in records] == [
        "run_started",
        "run_completed",
    ]
    assert [record.sequence for record in second_page] == [2]
    assert store.get_projection(snapshot.run_id, "run_result") is not None
    assert store.load_blob(blob_ref) == ("application/json", b'{"answer":"4"}')


def test_mongodb_store_backfills_a_missing_projection(
    mongodb_store_backend: MongoDbRunStore,
) -> None:
    store = mongodb_store_backend
    snapshot = _snapshot()

    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store._db()["run_projections"].delete_one(
        {"run_id": snapshot.run_id, "projection_name": "run_result"}
    )

    projection = store.get_projection(snapshot.run_id, "run_result")

    assert isinstance(projection, dict)
    assert projection["status"] == "running"
