from __future__ import annotations

from pathlib import Path

import pytest

from themis import (
    Experiment,
    InMemoryRunStore,
    RunResult,
    RunStore,
    RunSnapshot,
    RunStatus,
    RuntimeConfig,
    SqliteRunStore,
    get_run_snapshot,
    get_evaluation_execution,
    get_execution_state,
    sqlite_store,
)
from themis.core.case_refs import case_key_for
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import EvaluationCompletedEvent, RunStartedEvent
from themis.core.models import Case, Dataset
from tests.release import CURRENT_VERSION


def test_root_package_exports_public_symbols() -> None:
    from themis import (
        Experiment,
        Reporter,
        RunResult,
        RunSnapshot,
        RunStatus,
        RuntimeConfig,
        StatsEngine,
        evaluate_async,
        export_evaluation_bundle,
        export_generation_bundle,
        get_run_snapshot,
        get_evaluation_execution,
        get_execution_state,
        import_evaluation_bundle,
        import_generation_bundle,
        quickcheck,
        snapshot_report,
        sqlite_store,
    )

    assert Experiment is not None
    assert Reporter is not None
    assert RunResult is not None
    assert RunSnapshot is not None
    assert RunStatus is not None
    assert RuntimeConfig is not None
    assert StatsEngine is not None
    assert evaluate_async is not None
    assert export_evaluation_bundle is not None
    assert export_generation_bundle is not None
    assert import_evaluation_bundle is not None
    assert import_generation_bundle is not None
    assert get_run_snapshot is not None
    assert get_execution_state is not None
    assert get_evaluation_execution is not None
    assert quickcheck is not None
    assert snapshot_report is not None
    assert sqlite_store is not None


def test_public_surface_compiles_and_persists_runs(tmp_path) -> None:
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
        themis_version=CURRENT_VERSION,
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


def test_public_surface_runs_experiment_end_to_end() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
    )

    result = experiment.run(runtime=RuntimeConfig(max_concurrent_tasks=4))

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.COMPLETED


def test_package_includes_py_typed_marker() -> None:
    import themis

    assert Path(themis.__file__).with_name("py.typed").is_file()


def test_public_inspection_helpers_return_execution_state_and_evaluation_execution() -> (
    None
):
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "scores": [{"metric_id": "metric/judge", "value": 1.0}],
                "trace": {"trace_id": "trace-1", "steps": []},
            },
        )
    )

    stored_snapshot = get_run_snapshot(store, snapshot.run_id)
    state = get_execution_state(store, snapshot.run_id)
    execution = get_evaluation_execution(
        store, snapshot.run_id, "case-1", "metric/judge"
    )

    assert stored_snapshot.run_id == snapshot.run_id
    assert state.run_id == snapshot.run_id
    assert execution is not None
    assert execution.execution_id == "execution-1"


def test_get_evaluation_execution_rejects_conflicting_case_key() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(case_id="case-1", input={"question": "2+2"}),
                    Case(case_id="case-2", input={"question": "3+3"}),
                ],
            )
        ],
        seeds=[7],
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "scores": [{"metric_id": "metric/judge", "value": 1.0}],
                "trace": {"trace_id": "trace-1", "steps": []},
            },
        )
    )

    with pytest.raises(ValueError, match="Conflicting case_id and case_key inputs"):
        get_evaluation_execution(
            store,
            snapshot.run_id,
            "case-1",
            "metric/judge",
            case_key=case_key_for("dataset-1", "case-2"),
        )


def test_get_evaluation_execution_does_not_ignore_supplied_case_key() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"question": "2+2"})],
            )
        ],
        seeds=[7],
    )
    snapshot = experiment.compile()
    store = InMemoryRunStore()

    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(
        EvaluationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-reduced",
            metric_id="metric/judge",
            execution={
                "execution_id": "execution-1",
                "subject_kind": "candidate_set",
                "scores": [{"metric_id": "metric/judge", "value": 1.0}],
                "trace": {"trace_id": "trace-1", "steps": []},
            },
        )
    )

    assert (
        get_evaluation_execution(
            store,
            snapshot.run_id,
            "case-1",
            "metric/judge",
            case_key="unknown-case-key",
        )
        is None
    )
