from __future__ import annotations

from pathlib import Path

from themis import (
    Experiment,
    InMemoryRunStore,
    RunStore,
    RunSnapshot,
    SqliteRunStore,
    sqlite_store,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunStartedEvent
from themis.core.models import Case, Dataset


def test_root_package_exports_public_symbols() -> None:
    from themis import Experiment, RunSnapshot, sqlite_store

    assert Experiment is not None
    assert RunSnapshot is not None
    assert sqlite_store is not None


def test_public_surface_compiles_and_persists_runs(tmp_path) -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="generator/demo",
            candidate_policy={"num_samples": 1},
            reducer="reducer/demo",
        ),
        evaluation=EvaluationConfig(
            metrics=["metric/demo"],
            parsers=["parser/demo"],
            judge_config={"panel_size": 1},
        ),
        storage=StorageConfig(
            store="sqlite",
            parameters={"path": str(tmp_path / "run_store.sqlite3")},
        ),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output="4",
                    )
                ],
                revision="r1",
            )
        ],
        seeds=[7],
        environment_metadata={"env": "test"},
        themis_version="4.0.0a0",
        python_version="3.12.9",
        platform="macos",
    )

    snapshot = experiment.compile()
    memory_store = InMemoryRunStore()
    sqlite = sqlite_store(tmp_path / "run_store.sqlite3")

    assert isinstance(snapshot, RunSnapshot)
    assert isinstance(memory_store, RunStore)
    assert isinstance(sqlite, SqliteRunStore)

    memory_store.initialize()
    memory_store.persist_snapshot(snapshot)
    memory_store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    sqlite.initialize()
    sqlite.persist_snapshot(snapshot)
    sqlite.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    assert memory_store.resume(snapshot.run_id) is not None
    assert sqlite.resume(snapshot.run_id) is not None


def test_package_includes_py_typed_marker() -> None:
    import themis

    assert Path(themis.__file__).with_name("py.typed").is_file()
