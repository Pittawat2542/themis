from __future__ import annotations

import pytest

from themis.core.builtins import (
    resolve_generator_component,
    resolve_metric_component,
    resolve_parser_component,
    resolve_reducer_component,
    resolve_selector_component,
)
from themis.core.contexts import (
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.models import Case, GenerationResult, ParsedOutput, ScoreError
from themis.core.protocols import CandidateSelector, Generator, Parser, PureMetric
from themis.core.workflows import JudgeResponse


class ChoosingJudgeModel:
    component_id = "judge/custom"
    version = "1.0"

    def __init__(self, response: str) -> None:
        self.response = response

    def fingerprint(self) -> str:
        return "judge-custom-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=self.response,
        )


@pytest.mark.asyncio
async def test_builtin_generator_component_is_executable() -> None:
    generator = resolve_generator_component("builtin/demo_generator")
    case = Case(
        case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
    )

    result = await generator.generate(
        case, GenerateContext(run_id="run-1", case_id="case-1", seed=7)
    )

    assert isinstance(generator, Generator)
    assert isinstance(result, GenerationResult)
    assert result.candidate_id == "case-1-candidate-7"
    assert result.final_output == {"answer": "4"}
    assert result.token_usage == {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.mark.asyncio
async def test_builtin_reducer_parser_and_metric_components_are_executable() -> None:
    reducer = resolve_reducer_component("builtin/majority_vote")
    parser = resolve_parser_component("builtin/json_identity")
    metric = resolve_metric_component("builtin/exact_match")
    case = Case(
        case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
    )
    candidates = [
        GenerationResult(
            candidate_id="case-1-candidate-0", final_output={"answer": "4"}
        ),
        GenerationResult(
            candidate_id="case-1-candidate-1", final_output={"answer": "4"}
        ),
    ]

    reduced = await reducer.reduce(
        candidates,
        ReduceContext(
            run_id="run-1",
            case_id="case-1",
            candidate_ids=[candidate.candidate_id for candidate in candidates],
            seed=7,
        ),
    )
    parsed = parser.parse(
        reduced,
        ParseContext(
            run_id="run-1", case_id="case-1", candidate_id=reduced.candidate_id
        ),
    )
    assert isinstance(metric, PureMetric)
    score = metric.score(
        parsed,
        case,
        ScoreContext(run_id="run-1", case=case, parsed_output=parsed, seed=7),
    )

    assert reduced.source_candidate_ids == ["case-1-candidate-0", "case-1-candidate-1"]
    assert reduced.final_output == {"answer": "4"}
    assert isinstance(parser, Parser)
    assert parsed == ParsedOutput(value={"answer": "4"}, format="json")
    assert not isinstance(score, ScoreError)
    assert score.metric_id == "builtin/exact_match"
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_builtin_selector_component_is_executable() -> None:
    selector = resolve_selector_component("builtin/best_of_n")
    candidates = [
        GenerationResult(
            candidate_id="case-1-candidate-0", final_output={"answer": "4"}
        ),
        GenerationResult(
            candidate_id="case-1-candidate-1", final_output={"answer": "5"}
        ),
    ]

    selected = await selector.select(
        candidates,
        SelectContext(
            run_id="run-1",
            case_id="case-1",
            candidate_ids=[candidate.candidate_id for candidate in candidates],
            seed=7,
            judge_models=[ChoosingJudgeModel("B")],
        ),
    )

    assert isinstance(selector, CandidateSelector)
    assert [candidate.candidate_id for candidate in selected] == ["case-1-candidate-1"]


def test_runtime_component_resolvers_preserve_custom_objects() -> None:
    class CustomGenerator:
        component_id = "generator/custom"
        version = "1.0"

        def fingerprint(self) -> str:
            return "custom-generator"

        async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
            del case, ctx
            return GenerationResult(candidate_id="custom", final_output="ok")

    custom = CustomGenerator()

    assert resolve_generator_component(custom) is custom
