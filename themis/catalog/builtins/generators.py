"""Builtin generators and judge models."""

from __future__ import annotations

from themis.core.contexts import GenerateContext
from themis.core.models import Case, GenerationResult, Message
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
