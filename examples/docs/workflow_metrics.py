from __future__ import annotations

from themis import Experiment, InMemoryRunStore, get_evaluation_execution
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Execute builtin workflow-backed metrics together."""

    store = InMemoryRunStore()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=[
                "builtin/llm_rubric",
                "builtin/panel_of_judges",
                "builtin/majority_vote_judge",
                "builtin/pairwise_judge",
            ],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_overrides={"rubric": "pass if the answer is correct"},
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
    execution = get_evaluation_execution(
        store, result.run_id, "case-1", "builtin/llm_rubric"
    )
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].scores],
        "judge_calls": 0 if execution is None else len(execution.judge_calls),
    }


if __name__ == "__main__":
    print(run_example())
