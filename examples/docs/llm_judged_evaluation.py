from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis.analysis import get_evaluation_execution
from themis import Evaluation, Generation
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Run a workflow-backed metric with builtin demo judges."""

    store = memory_store()
    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator",
            samples=1,
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=["builtin/llm_rubric"],
            parser="builtin/json_identity",
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_options={"rubric": "pass if the answer is correct"},
        ),
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
        seeds=[7],
    )

    result = experiment.run(store=store)
    execution = get_evaluation_execution(
        store, result.run_id, "case-1", "builtin/llm_rubric"
    )
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "judge_calls": 0 if execution is None else len(execution.judge_calls),
        "score_count": 0 if execution is None else len(execution.metric_results),
    }


if __name__ == "__main__":
    print(run_example())
