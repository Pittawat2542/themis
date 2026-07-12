from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis.analysis import get_evaluation_execution
from themis import Evaluation, Generation
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Execute builtin workflow-backed metrics together."""

    store = memory_store()
    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator",
            samples=2,
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=[
                "builtin/llm_rubric",
                "builtin/panel_of_judges",
                "builtin/majority_vote_judge",
                "builtin/pairwise_judge",
            ],
            parser="builtin/json_identity",
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_options={"rubric": "pass if the answer is correct"},
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
    execution = get_evaluation_execution(
        store, result.run_id, "case-1", "builtin/llm_rubric"
    )
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].metric_results],
        "judge_calls": 0 if execution is None else len(execution.judge_calls),
    }


if __name__ == "__main__":
    print(run_example())
