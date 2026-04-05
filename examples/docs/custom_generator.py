from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset, GenerationResult


class CustomGenerator:
    """Small example generator that satisfies the Generator protocol."""

    component_id = "generator/custom_example"
    version = "1.0"

    def fingerprint(self) -> str:
        return "custom-example-generator"

    async def generate(self, case: Case, ctx: object) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate",
            final_output={"answer": "4"},
        )


def run_example() -> dict[str, object]:
    """Execute an experiment with a custom generator instance."""

    experiment = Experiment(
        generation=GenerationConfig(generator=CustomGenerator()),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]
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
    )
    result = experiment.run()
    return {"run_id": result.run_id, "status": result.status.value}


if __name__ == "__main__":
    print(run_example())
