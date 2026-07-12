from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis import Evaluation, Generation
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Run a multi-candidate evaluation with mixed metrics."""

    store = memory_store()
    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator",
            samples=2,
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=[
                "builtin/exact_match",
                "builtin/llm_rubric",
                "builtin/pairwise_judge",
            ],
            parser="builtin/json_identity",
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_options={"rubric": "prefer correct and concise answers"},
        ),
        datasets=[
            inline_dataset_source(
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
        "score_ids": [score.metric_id for score in case_result.metric_results],
    }


if __name__ == "__main__":
    print(run_example())
