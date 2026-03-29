from __future__ import annotations

from themis import InMemoryRunStore, RunStatus
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.events import RunStartedEvent
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

    result = experiment.run()

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
