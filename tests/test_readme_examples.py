from __future__ import annotations

from themis import Case, Dataset, Evaluation, Experiment, Generation
from themis.analysis import get_execution_state
from themis.storage import memory_store


def test_readme_quick_start_runs_through_the_public_api() -> None:
    store = memory_store()
    experiment = Experiment(
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
        generation=Generation(
            generator="builtin/demo_generator",
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"],
            parser="builtin/json_identity",
        ),
    )

    result = experiment.run(store=store)

    assert result.status.value == "completed"
    assert get_execution_state(store, result.run_id).status.value == "completed"
