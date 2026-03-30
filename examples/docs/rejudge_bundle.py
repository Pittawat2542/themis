from __future__ import annotations

from themis import (
    Experiment,
    InMemoryRunStore,
    export_evaluation_bundle,
    export_generation_bundle,
    import_evaluation_bundle,
    import_generation_bundle,
)
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Export bundle artifacts, import them into another store, and rejudge in place."""

    source_store = InMemoryRunStore()
    target_store = InMemoryRunStore()
    source_store.initialize()
    target_store.initialize()

    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/llm_rubric"],
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
        seeds=[7],
    )
    initial = experiment.run(store=source_store)
    import_generation_bundle(target_store, export_generation_bundle(source_store, initial.run_id))
    import_evaluation_bundle(target_store, export_evaluation_bundle(source_store, initial.run_id))
    rejudged = experiment.rejudge(store=source_store)
    return {
        "run_id": initial.run_id,
        "rejudged_run_id": rejudged.run_id,
        "imported": target_store.resume(initial.run_id) is not None,
    }


if __name__ == "__main__":
    print(run_example())
