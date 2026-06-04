from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.dataset_sources import inline_dataset_source
from themis.core.models import Case, Dataset


def run_example() -> dict[str, object]:
    """Execute builtin pure metrics together."""

    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator", reducer="builtin/majority_vote"
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match", "builtin/f1", "builtin/bleu"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
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
    )
    result = experiment.run()
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].metric_results],
    }


if __name__ == "__main__":
    print(run_example())
