"""Executable builtin runtime components for Themis Phase 2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.models import Case, GenerationResult, Message, ParsedOutput, ReducedCandidate, Score
from themis.core.protocols import CandidateReducer, Generator, JudgeModel, Parser, PureMetric
from themis.core.workflows import JudgeResponse


class DemoGenerator:
    component_id = "generator/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-generator-demo-fingerprint"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        answer = case.expected_output if case.expected_output is not None else case.input
        candidate_suffix = ctx.seed if ctx.seed is not None else 0
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{candidate_suffix}",
            final_output=answer,
            conversation=[Message(role="assistant", content=answer)],
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_ms=1.0,
        )


class DemoReducer:
    component_id = "reducer/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-reducer-demo-fingerprint"

    def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
            metadata={"strategy": "first_candidate"},
        )


class DemoParser:
    component_id = "parser/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-parser-demo-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")


class DemoMetric:
    component_id = "metric/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-metric-demo-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        return Score(metric_id=self.component_id, value=float(parsed.value == case.expected_output))


class DemoJudgeModel:
    component_id = "judge/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-judge-demo-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response="pass" if prompt else "fail",
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_ms=1.0,
        )


_BUILTIN_GENERATORS: dict[str, Generator] = {"generator/demo": DemoGenerator()}
_BUILTIN_REDUCERS: dict[str, CandidateReducer] = {"reducer/demo": DemoReducer()}
_BUILTIN_PARSERS: dict[str, Parser] = {"parser/demo": DemoParser()}
_BUILTIN_METRICS: dict[str, PureMetric] = {"metric/demo": DemoMetric()}
_BUILTIN_JUDGE_MODELS: dict[str, JudgeModel] = {"judge/demo": DemoJudgeModel()}


def _resolve(mapping: Mapping[str, object], value: object) -> object:
    if isinstance(value, str):
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"Unknown builtin component: {value}") from exc
    return value


def resolve_generator_component(value: object) -> Generator:
    return cast(Generator, _resolve(_BUILTIN_GENERATORS, value))


def resolve_reducer_component(value: object) -> CandidateReducer:
    return cast(CandidateReducer, _resolve(_BUILTIN_REDUCERS, value))


def resolve_parser_component(value: object) -> Parser:
    return cast(Parser, _resolve(_BUILTIN_PARSERS, value))


def resolve_metric_component(value: object) -> PureMetric:
    return cast(PureMetric, _resolve(_BUILTIN_METRICS, value))


def resolve_judge_model_component(value: object) -> JudgeModel:
    return cast(JudgeModel, _resolve(_BUILTIN_JUDGE_MODELS, value))
