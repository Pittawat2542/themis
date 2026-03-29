from __future__ import annotations

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import create_run_store
from themis.core.stores.mongodb import MongoDbRunStore, mongodb_store
from tests.core.store_fakes import fake_pymongo_module


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="mongodb"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                revision="r1",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output="4")],
            )
        ],
        seeds=[7],
    )
    return experiment.compile()


def test_mongodb_store_persists_events_projections_and_blobs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("themis.core.stores.mongodb.importlib.import_module", lambda name: fake_pymongo_module())
    blob_root = tmp_path / "mongodb-blobs"
    store = mongodb_store("mongodb://example", "themis_test", blob_root)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    blob_ref = store.store_blob(b'{"answer":"4"}', "application/json")

    assert isinstance(store, MongoDbRunStore)
    assert store.resume(snapshot.run_id) is not None
    assert [event.event_type for event in store.query_events(snapshot.run_id)] == ["run_started", "run_completed"]
    assert store.get_projection(snapshot.run_id, "run_result") is not None
    assert store.load_blob(blob_ref) == ("application/json", b'{"answer":"4"}')


def test_store_factory_can_build_mongodb_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("themis.core.stores.mongodb.importlib.import_module", lambda name: fake_pymongo_module())

    store = create_run_store(
        StorageConfig(
            store="mongodb",
            parameters={
                "url": "mongodb://example",
                "database": "themis_test",
                "blob_root": str(tmp_path / "mongodb-blobs"),
            },
        )
    )

    assert isinstance(store, MongoDbRunStore)
