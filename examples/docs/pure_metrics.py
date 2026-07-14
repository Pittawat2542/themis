from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis import Evaluation, Generation
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Execute builtin pure metrics together."""

    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator", reducer="builtin/majority_vote"
        ),
        evaluation=Evaluation(
            metrics=[
                "builtin/exact_match",
                "builtin/f1",
                "builtin/bleu",
                "builtin/rouge_l",
            ],
            parser="builtin/json_identity",
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
    )
    result = experiment.run(store=memory_store())
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].metric_results],
    }


if __name__ == "__main__":
    print(run_example())
