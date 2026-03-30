from __future__ import annotations

from themis import Experiment, InMemoryRunStore, RunStatus, evaluate
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def test_evaluate_runs_one_off_experiment_with_same_run_id_as_explicit_experiment() -> None:
    datasets = [
        Dataset(
            dataset_id="dataset-1",
            cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            revision="r1",
        )
    ]
    generation = GenerationConfig(
        generator="builtin/demo_generator",
        candidate_policy={"num_samples": 1},
        reducer="builtin/majority_vote",
    )
    evaluation = EvaluationConfig(
        metrics=["builtin/exact_match"],
        parsers=["builtin/json_identity"],
    )
    storage = StorageConfig(store="memory")

    explicit = Experiment(
        generation=generation,
        evaluation=evaluation,
        storage=storage,
        datasets=datasets,
        seeds=[7],
    )
    store = InMemoryRunStore()

    result = evaluate(
        generation=generation,
        evaluation=evaluation,
        storage=storage,
        datasets=datasets,
        seeds=[7],
        store=store,
    )

    assert result.status is RunStatus.COMPLETED
    assert result.run_id == explicit.compile().run_id
