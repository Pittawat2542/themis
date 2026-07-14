from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis import Evaluation, Generation
from themis import Case, Dataset
from themis.components import ParseContext, ParsedOutput, ReducedCandidate


class AnswerStringParser:
    """Parser example that normalizes a JSON-like answer payload to a string."""

    component_id = "parser/answer_string"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-answer-string"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        value = candidate.final_output
        if isinstance(value, dict) and "answer" in value:
            return ParsedOutput(value=str(value["answer"]), format="string")
        return ParsedOutput(value=str(value), format="string")


def run_example() -> dict[str, object]:
    """Execute an experiment with a custom parser."""

    experiment = Experiment(
        generation=Generation(
            generator="builtin/demo_generator", reducer="builtin/majority_vote"
        ),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"], parser=AnswerStringParser()
        ),
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output="4",
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
