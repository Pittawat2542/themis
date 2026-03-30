from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, RuntimeConfig, StorageConfig
from themis.core.models import Case, Dataset
from themis.core.planner import Planner


def test_planner_estimate_returns_deterministic_counts_for_compiled_snapshot() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 3},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match", "builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}),
                    Case(case_id="case-2", input={"question": "3+3"}, expected_output={"answer": "6"}),
                ],
            )
        ],
    )

    estimate = Planner().estimate(experiment.compile())

    assert estimate.run_id == experiment.compile().run_id
    assert estimate.total_cases == 2
    assert estimate.candidate_count == 3
    assert estimate.planned_generation_tasks == 6
    assert estimate.planned_reduction_tasks == 2
    assert estimate.planned_parse_tasks == 2
    assert estimate.planned_score_tasks == 4


def test_runtime_submission_paths_do_not_affect_run_identity() -> None:
    default_runtime = Experiment(
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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
    )
    worker_paths = Experiment(
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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        runtime=RuntimeConfig(queue_root="runs/queue"),
    )
    batch_paths = Experiment(
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
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        runtime=RuntimeConfig(batch_root="runs/batch"),
    )

    assert default_runtime.compile().run_id == worker_paths.compile().run_id == batch_paths.compile().run_id
