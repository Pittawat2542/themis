from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Run a small end-to-end evaluation through an explicit experiment."""

    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
    result = experiment.run()
    return {"run_id": result.run_id, "status": result.status.value}


if __name__ == "__main__":
    print(run_example())
