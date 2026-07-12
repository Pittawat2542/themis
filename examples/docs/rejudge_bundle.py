from __future__ import annotations

from themis.storage import memory_store

from themis import (
    Experiment,
)
from themis.runtime import Stage
from themis.core.bundles import (
    export_evaluation_bundle,
    export_generation_bundle,
    import_evaluation_bundle,
    import_generation_bundle,
)
from themis import Evaluation, Generation
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset


def run_example() -> dict[str, object]:
    """Export bundle artifacts, import them into another store, and replay judge scoring in place."""

    source_store = memory_store()
    target_store = memory_store()
    source_store.initialize()
    target_store.initialize()

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
    initial = experiment.run(store=source_store)
    import_generation_bundle(
        target_store, export_generation_bundle(source_store, initial.run_id)
    )
    import_evaluation_bundle(
        target_store, export_evaluation_bundle(source_store, initial.run_id)
    )
    replayed = experiment.replay(from_stage=Stage.JUDGE, store=source_store)
    return {
        "run_id": initial.run_id,
        "replayed_run_id": replayed.run_id,
        "imported": target_store.resume(initial.run_id) is not None,
    }


if __name__ == "__main__":
    print(run_example())
