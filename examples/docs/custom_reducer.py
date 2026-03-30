from __future__ import annotations

from collections.abc import Mapping

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import GenerateContext, ReduceContext
from themis.core.models import Case, Dataset, GenerationResult, ReducedCandidate


class SeededGenerator:
    """Generator example that emits different answers per seed."""

    component_id = "generator/seeded_example"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-seeded-example"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        answer = "4" if (ctx.seed or 0) % 2 else "5"
        return GenerationResult(candidate_id=f"{case.case_id}-candidate-{ctx.seed or 0}", final_output={"answer": answer})


class PreferCorrectReducer:
    """Reducer example that picks the numerically smaller answer."""

    component_id = "reducer/prefer_correct"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-prefer-correct"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        winner = sorted(candidates, key=_answer_value)[0]
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=winner.final_output,
            metadata={"strategy": "prefer_numeric_minimum"},
        )


def _answer_value(candidate: GenerationResult) -> int:
    final_output = candidate.final_output
    if isinstance(final_output, Mapping) and "answer" in final_output:
        return int(str(final_output["answer"]))
    raise TypeError("Expected mapping final_output with an 'answer' key.")


def run_example() -> dict[str, object]:
    """Execute an experiment with a custom reducer."""

    experiment = Experiment(
        generation=GenerationConfig(
            generator=SeededGenerator(),
            candidate_policy={"num_samples": 2},
            reducer=PreferCorrectReducer(),
        ),
        evaluation=EvaluationConfig(metrics=["builtin/exact_match"], parsers=["builtin/json_identity"]),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
            )
        ],
        seeds=[7, 8],
    )
    result = experiment.run()
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].scores],
    }


if __name__ == "__main__":
    print(run_example())
