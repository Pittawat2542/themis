from __future__ import annotations

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import InMemoryRunStore, SqliteRunStore, create_run_store, register_store_backend


def _experiment(store: str, parameters: dict[str, object] | None = None) -> Experiment:
    return Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store=store, parameters=parameters or {}),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input="hi", expected_output="hi")])],
    )


def test_create_run_store_supports_builtin_backends(tmp_path) -> None:
    memory_store = create_run_store(StorageConfig(store="memory"))
    sqlite = create_run_store(
        StorageConfig(store="sqlite", parameters={"path": str(tmp_path / "run_store.sqlite3")})
    )

    assert isinstance(memory_store, InMemoryRunStore)
    assert isinstance(sqlite, SqliteRunStore)


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

    monkeypatch.setattr("themis.core.experiment.create_run_store", fake_create_run_store)

    store = _experiment("memory")._build_store()

    assert store is sentinel
    assert captured == [StorageConfig(store="memory", parameters={})]
