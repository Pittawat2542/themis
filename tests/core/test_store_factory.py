from __future__ import annotations

from themis.core.base import JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import (
    InMemoryRunStore,
    JsonlRunStore,
    MongoDbRunStore,
    PostgresRunStore,
    SqliteRunStore,
    create_run_store,
    register_store_backend,
)
from tests.core.store_fakes import fake_pymongo_module


def _experiment(
    store: str, parameters: dict[str, JSONValue] | None = None
) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(store=store, parameters=parameters or {}),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input="hi", expected_output="hi")],
            )
        ],
    )


def test_create_run_store_supports_builtin_backends(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "themis.core.stores.mongodb.importlib.import_module",
        lambda name: fake_pymongo_module(),
    )

    memory_store = create_run_store(StorageConfig(store="memory"))
    jsonl = create_run_store(
        StorageConfig(store="jsonl", parameters={"root": str(tmp_path / "jsonl-store")})
    )
    sqlite = create_run_store(
        StorageConfig(
            store="sqlite", parameters={"path": str(tmp_path / "run_store.sqlite3")}
        )
    )
    postgres = create_run_store(
        StorageConfig(
            store="postgres",
            parameters={
                "url": str(tmp_path / "postgres.sqlite3"),
                "blob_root": str(tmp_path / "postgres-blobs"),
            },
        )
    )
    mongodb = create_run_store(
        StorageConfig(
            store="mongodb",
            parameters={
                "url": "mongodb://example",
                "database": "themis_test",
                "blob_root": str(tmp_path / "mongodb-blobs"),
            },
        )
    )

    assert isinstance(memory_store, InMemoryRunStore)
    assert isinstance(jsonl, JsonlRunStore)
    assert isinstance(sqlite, SqliteRunStore)
    assert isinstance(postgres, PostgresRunStore)
    assert isinstance(mongodb, MongoDbRunStore)


def test_register_store_backend_allows_custom_builders() -> None:
    class DummyStore(InMemoryRunStore):
        pass

    register_store_backend("dummy", lambda config: DummyStore())

    store = create_run_store(StorageConfig(store="dummy"))

    assert isinstance(store, DummyStore)


def test_experiment_build_store_routes_through_store_factory(monkeypatch) -> None:
    captured: list[StorageConfig] = []
    sentinel = InMemoryRunStore()

    def fake_create_run_store(config: StorageConfig):
        captured.append(config)
        return sentinel

    monkeypatch.setattr(
        "themis.core.experiment.create_run_store", fake_create_run_store
    )

    store = _experiment("memory")._build_store()

    assert store is sentinel
    assert captured == [StorageConfig(store="memory", parameters={})]
