from __future__ import annotations

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores.mongodb import mongodb_store
from tests.core.store_fakes import FakeCollection, fake_pymongo_module


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
                        case_id="case-1", input={"question": "2+2"}, expected_output="4"
                    )
                ],
            )
        ],
        seeds=[7],
    )
    return experiment.compile()


def test_mongodb_persist_event_allocates_sequences_without_querying_events(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "themis.core.stores.mongodb.importlib.import_module",
        lambda name: fake_pymongo_module(),
    )
    store = mongodb_store(
        "mongodb://example", "themis_test", tmp_path / "mongodb-blobs"
    )
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.query_events = lambda run_id: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("persist_event should not query existing events")
    )

    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))

    rows = store._db()["run_events"].find({"run_id": snapshot.run_id})

    assert [row["sequence"] for row in rows] == [1, 2]


def test_fake_collection_find_one_and_update_honors_return_document() -> None:
    collection = FakeCollection()

    assert (
        collection.find_one_and_update(
            {"run_id": "run-1"},
            {"$inc": {"next_sequence": 1}},
            upsert=True,
            return_document=False,
        )
        is None
    )
    assert collection.find_one({"run_id": "run-1"}) == {
        "run_id": "run-1",
        "next_sequence": 1,
    }

    after = collection.find_one_and_update(
        {"run_id": "run-1"},
        {"$inc": {"next_sequence": 1}},
        return_document=True,
    )
    assert after == {"run_id": "run-1", "next_sequence": 2}

    before = collection.find_one_and_update(
        {"run_id": "run-1"},
        {"$inc": {"next_sequence": 1}},
        return_document=False,
    )
    assert before == {"run_id": "run-1", "next_sequence": 2}
    assert collection.find_one({"run_id": "run-1"}) == {
        "run_id": "run-1",
        "next_sequence": 3,
    }

    with pytest.raises(
        ValueError,
        match="return_document must be ReturnDocument.BEFORE or ReturnDocument.AFTER",
    ):
        collection.find_one_and_update(
            {"run_id": "run-1"},
            {"$inc": {"next_sequence": 1}},
            return_document="after",
        )
