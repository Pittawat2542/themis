from __future__ import annotations

import types

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import create_run_store
from themis.core.stores.mongodb import MongoDbRunStore, mongodb_store


class _FakeCollection:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def replace_one(self, query: dict[str, object], document: dict[str, object], upsert: bool = False) -> None:
        for index, row in enumerate(self.rows):
            if all(row.get(key) == value for key, value in query.items()):
                self.rows[index] = dict(document)
                return
        if upsert:
            self.rows.append(dict(document))

    def insert_one(self, document: dict[str, object]) -> None:
        self.rows.append(dict(document))

    def find(self, query: dict[str, object]) -> list[dict[str, object]]:
        return [row for row in self.rows if all(row.get(key) == value for key, value in query.items())]

    def find_one(self, query: dict[str, object]) -> dict[str, object] | None:
        for row in self.rows:
            if all(row.get(key) == value for key, value in query.items()):
                return row
        return None


class _FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


class _FakeMongoClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.databases: dict[str, _FakeDatabase] = {}

    def __getitem__(self, name: str) -> _FakeDatabase:
        if name not in self.databases:
            self.databases[name] = _FakeDatabase()
        return self.databases[name]


def _fake_pymongo_module():
    module = types.SimpleNamespace()
    module.MongoClient = lambda url: _FakeMongoClient(url)
    return module


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
    monkeypatch.setattr("themis.core.stores.mongodb.importlib.import_module", lambda name: _fake_pymongo_module())
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
    monkeypatch.setattr("themis.core.stores.mongodb.importlib.import_module", lambda name: _fake_pymongo_module())

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
