from __future__ import annotations

from themis import Experiment, InMemoryRunStore
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Run a multi-candidate evaluation with mixed metrics."""

    store = InMemoryRunStore()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match", "builtin/llm_rubric", "builtin/pairwise_judge"],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_overrides={"rubric": "prefer correct and concise answers"},
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7, 11],
    )

    result = experiment.run(store=store)
    case_result = result.cases[0]
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "generated_candidates": len(case_result.generated_candidates),
        "score_ids": [score.metric_id for score in case_result.scores],
    }


if __name__ == "__main__":
    print(run_example())
