from __future__ import annotations

from themis import Experiment, RunOptions
from themis.storage import memory_store
from themis import Evaluation, Generation
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Compile and run an explicit Experiment definition."""

    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator",
            samples=1,
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"],
            parser="builtin/json_identity",
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
        seeds=[7],
    )
    snapshot = experiment.compile()
    result = experiment.run(
        store=memory_store(), options=RunOptions(max_concurrency=4)
    )
    return {"run_id": snapshot.run_id, "status": result.status.value}


if __name__ == "__main__":
    print(run_example())
