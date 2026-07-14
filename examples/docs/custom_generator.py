from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis import Evaluation, Generation
from themis import Case, Dataset
from themis.components import Candidate


class CustomGenerator:
    """Small example generator that satisfies the Generator protocol."""

    component_id = "generator/custom_example"
    version = "1.0"

    def fingerprint(self) -> str:
        return "custom-example-generator"

    async def generate(self, case: Case, ctx: object) -> Candidate:
        del ctx
        return Candidate(
            candidate_id=f"{case.case_id}-candidate",
            final_output={"answer": "4"},
        )


def run_example() -> dict[str, object]:
    """Execute an experiment with a custom generator instance."""

    experiment = Experiment(
        generation=Generation(generator=CustomGenerator()),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"], parser="builtin/json_identity"
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
    return {"run_id": result.run_id, "status": result.status.value}


if __name__ == "__main__":
    print(run_example())
