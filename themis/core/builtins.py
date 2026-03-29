"""Executable builtin runtime components for Themis Phase 4."""

from __future__ import annotations

from typing import cast

from themis.catalog.registry import load_component
from themis.core.contexts import GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.models import Case, GenerationResult, Message, ParsedOutput, ReducedCandidate, Score
from themis.core.protocols import (
    CandidateReducer,
    Generator,
    JudgeModel,
    LLMMetric,
    Parser,
    PureMetric,
    SelectionMetric,
    TraceMetric,
)
from themis.core.workflows import JudgeResponse


class DemoGenerator:
    component_id = "builtin/demo_generator"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-demo-generator-fingerprint"

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
    component_id = "builtin/majority_vote"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-majority-vote-fingerprint"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        winning_candidate = max(
            candidates,
            key=lambda candidate: (
                sum(1 for peer in candidates if peer.final_output == candidate.final_output),
                -candidates.index(candidate),
            ),
        )
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=winning_candidate.final_output,
            metadata={"strategy": "majority_vote"},
        )


class DemoParser:
    component_id = "builtin/json_identity"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-json-identity-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")


class DemoMetric:
    component_id = "builtin/exact_match"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-exact-match-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        return Score(metric_id=self.component_id, value=float(parsed.value == case.expected_output))


class DemoJudgeModel:
    component_id = "builtin/demo_judge"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-demo-judge-fingerprint"

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
BuiltinMetric = PureMetric | LLMMetric | SelectionMetric | TraceMetric


def _resolve(value: object, *, kind: str) -> object:
    if isinstance(value, str):
        return load_component(value, kind=kind)
    return value


def resolve_generator_component(value: object) -> Generator:
    return cast(Generator, _resolve(value, kind="generator"))


def resolve_reducer_component(value: object) -> CandidateReducer:
    return cast(CandidateReducer, _resolve(value, kind="reducer"))


def resolve_parser_component(value: object) -> Parser:
    return cast(Parser, _resolve(value, kind="parser"))


def resolve_metric_component(value: object) -> BuiltinMetric:
    return cast(BuiltinMetric, _resolve(value, kind="metric"))


def resolve_judge_model_component(value: object) -> JudgeModel:
    return cast(JudgeModel, _resolve(value, kind="judge_model"))
