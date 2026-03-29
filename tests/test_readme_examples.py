from __future__ import annotations

from themis import InMemoryRunStore, RunStatus, RuntimeConfig
from themis.core import (
    export_evaluation_bundle,
    export_generation_bundle,
    import_evaluation_bundle,
    import_generation_bundle,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import EvaluationCompletedEvent, GenerationCompletedEvent, RunStartedEvent
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult


class CustomGenerator:
    component_id = "generator/custom"
    version = "1.0"

    def fingerprint(self) -> str:
        return "custom-generator-fingerprint"

    async def generate(self, case: Case, ctx: object) -> GenerationResult:
        del ctx
        return GenerationResult(candidate_id=f"{case.case_id}-candidate", final_output={"answer": "4"})


def test_readme_builtin_component_example_compiles_and_persists() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/demo", reducer="reducer/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    )
    store = InMemoryRunStore()

    snapshot = experiment.compile()
    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    assert store.resume(snapshot.run_id) is not None


def test_readme_builtin_component_example_runs_end_to_end() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/demo", reducer="reducer/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"}, expected_output={"answer": "4"})])],
    )

    result = experiment.run(
        runtime=RuntimeConfig(
            max_concurrent_tasks=8,
            stage_concurrency={"generation": 4},
        )
    )

    assert result.status is RunStatus.COMPLETED


def test_readme_custom_component_example_compiles_and_persists() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator=CustomGenerator()),
        evaluation=EvaluationConfig(),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    )
    store = InMemoryRunStore()

    snapshot = experiment.compile()
    store.initialize()
    store.persist_snapshot(snapshot)
    store.persist_event(RunStartedEvent(run_id=snapshot.run_id))

    assert store.resume(snapshot.run_id) is not None


def test_readme_custom_component_example_runs_end_to_end() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator=CustomGenerator()),
        evaluation=EvaluationConfig(),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input={"q": "2+2"})])],
    )

    result = experiment.run()

    assert result.status is RunStatus.COMPLETED


def test_readme_generation_bundle_example_round_trips() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/demo", reducer="reducer/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
    )
    source_store = InMemoryRunStore()
    source_store.initialize()
    snapshot = experiment.compile()
    source_store.persist_snapshot(snapshot)
    source_store.persist_event(RunStartedEvent(run_id=snapshot.run_id))
    source_store.persist_event(
        GenerationCompletedEvent(
            run_id=snapshot.run_id,
            case_id="case-1",
            candidate_id="case-1-candidate-0",
            candidate_index=0,
            result={
                "candidate_id": "case-1-candidate-0",
                "final_output": {"answer": "4"},
            },
        )
    )

    target_store = InMemoryRunStore()
    target_store.initialize()
    bundle = export_generation_bundle(source_store, snapshot.run_id)
    import_generation_bundle(target_store, bundle)

    assert target_store.resume(snapshot.run_id) is not None


def test_readme_evaluation_bundle_example_round_trips() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator="generator/demo", reducer="reducer/demo"),
        evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[Case(case_id="case-1", input={"q": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
    )
    source_store = InMemoryRunStore()
    source_store.initialize()
    snapshot = experiment.compile()
    source_store.persist_snapshot(snapshot)
    source_store.persist_event(
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

    target_store = InMemoryRunStore()
    target_store.initialize()
    bundle = export_evaluation_bundle(source_store, snapshot.run_id)
    import_evaluation_bundle(target_store, bundle)

    assert target_store.resume(snapshot.run_id) is not None
