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
        EvaluationGraph,
        EvaluationStep,
        CodeExecutionLimits,
        CodeExecutionRequest,
        CodeExecutionResult,
        CodeExecutionStatus,
        DockerExecutionBackend,
        ExecutionBackend,
        ExecutionResourcePlan,
        ExecutionRequest,
        FilesystemExecutionBackend,
        GraphRuntime,
        GraphRunResult,
        InMemoryExecutionBackend,
        LocalSubprocessExecutionBackend,
        PairwiseComparisonReport,
        PairwiseMetricClaim,
        ProviderCallCompletedEvent,
        ProviderCallFailedEvent,
        ProviderCallStartedEvent,
        ProviderTelemetry,
        QueueExecutionBackend,
        StepInput,
        StepOutput,
        DatasetSourceSpec,
        Reporter,
        ReporterProtocol,
        RunLineage,
        RunQuery,
        RunRecord,
        RunSnapshot,
        RunStatus,
        RuntimeConfig,
        StageTelemetry,
        SuiteCoverageSummary,
        StatsEngine,
        available_reporters,
        create_reporter,
        get_case_audit,
        get_run_snapshot,
        get_telemetry_summary,
        get_evaluation_execution,
        get_execution_state,
        get_preset,
        list_presets,
        apply_preset,
        sqlite_store,
        register_reporter,
    )

    assert EvaluationGraph is not None
    assert EvaluationStep is not None
    assert CodeExecutionLimits is not None
    assert CodeExecutionRequest is not None
    assert CodeExecutionResult is not None
    assert CodeExecutionStatus is not None
    assert DockerExecutionBackend is not None
    assert ExecutionBackend is not None
    assert ExecutionResourcePlan is not None
    assert ExecutionRequest is not None
    assert FilesystemExecutionBackend is not None
    assert GraphRuntime is not None
    assert GraphRunResult is not None
    assert InMemoryExecutionBackend is not None
    assert LocalSubprocessExecutionBackend is not None
    assert PairwiseComparisonReport is not None
    assert PairwiseMetricClaim is not None
    assert ProviderCallCompletedEvent is not None
    assert ProviderCallFailedEvent is not None
    assert ProviderCallStartedEvent is not None
    assert ProviderTelemetry is not None
    assert QueueExecutionBackend is not None
    assert StepInput is not None
    assert StepOutput is not None
    assert DatasetSourceSpec is not None
    assert Reporter is not None
    assert ReporterProtocol is not None
    assert RunLineage is not None
    assert RunQuery is not None
    assert RunRecord is not None
    assert RunSnapshot is not None
    assert RunStatus is not None
    assert RuntimeConfig is not None
    assert StageTelemetry is not None
    assert SuiteCoverageSummary is not None
    assert StatsEngine is not None
    assert available_reporters is not None
    assert create_reporter is not None
    assert get_case_audit is not None
    assert get_run_snapshot is not None
    assert get_telemetry_summary is not None
    assert get_execution_state is not None
    assert get_evaluation_execution is not None
    assert get_preset is not None
    assert list_presets is not None
    assert apply_preset is not None
    assert sqlite_store is not None
    assert register_reporter is not None


def test_root_package_does_not_export_legacy_convenience_surface() -> None:
    import themis

    for name in (
        "evaluate",
        "evaluate_async",
        "quickcheck",
        "snapshot_report",
        "export_generation_bundle",
        "export_evaluation_bundle",
        "import_generation_bundle",
        "import_evaluation_bundle",
    ):
        assert name not in themis.__all__
        assert not hasattr(themis, name)


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
            target="sqlite",
            kwargs={"path": str(tmp_path / "run_store.sqlite3")},
        ),
        dataset_sources=[
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
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
                "metric_results": [{"metric_id": "metric/judge", "value": 1.0}],
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
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
                "metric_results": [{"metric_id": "metric/judge", "value": 1.0}],
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
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
                "metric_results": [{"metric_id": "metric/judge", "value": 1.0}],
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
