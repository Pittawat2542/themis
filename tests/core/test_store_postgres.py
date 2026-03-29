from __future__ import annotations

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.stores import create_run_store
from themis.core.stores.postgres import PostgresRunStore, postgres_store
from tests.core.store_fakes import fake_psycopg_module


def _snapshot():
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="postgres"),
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


def test_postgres_store_persists_events_projections_and_blobs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("themis.core.stores.postgres.importlib.import_module", lambda name: fake_psycopg_module())
    database_path = tmp_path / "postgres.sqlite3"
    blob_root = tmp_path / "postgres-blobs"
    store = postgres_store(str(database_path), blob_root)
    snapshot = _snapshot()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    store.persist_event(RunCompletedEvent(run_id=snapshot.run_id))
    blob_ref = store.store_blob(b'{"answer":"4"}', "application/json")

    assert isinstance(store, PostgresRunStore)
    assert store.resume(snapshot.run_id) is not None
    assert [event.event_type for event in store.query_events(snapshot.run_id)] == ["run_started", "run_completed"]
    assert store.get_projection(snapshot.run_id, "run_result") is not None
    assert store.load_blob(blob_ref) == ("application/json", b'{"answer":"4"}')


def test_store_factory_can_build_postgres_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("themis.core.stores.postgres.importlib.import_module", lambda name: fake_psycopg_module())

    store = create_run_store(
        StorageConfig(
            store="postgres",
            parameters={
                "url": str(tmp_path / "postgres.sqlite3"),
                "blob_root": str(tmp_path / "postgres-blobs"),
            },
        )
    )

    assert isinstance(store, PostgresRunStore)
