"""Builtin candidate selectors."""

from __future__ import annotations

from themis.core.contexts import SelectContext
from themis.core.models import GenerationResult
from themis.core.protocols import JudgeModel


class BestOfNSelector:
    component_id = "builtin/best_of_n"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-best-of-n-fingerprint"

    async def select(
        self, candidates: list[GenerationResult], ctx: SelectContext
    ) -> list[GenerationResult]:
        if len(candidates) <= 1 or not ctx.judge_models:
            return candidates[:1]
        winner = await _select_best_candidate(candidates, ctx.judge_models)
        return [winner]


async def _select_best_candidate(
    candidates: list[GenerationResult],
    judge_models: list[JudgeModel],
) -> GenerationResult:
    winner = candidates[0]
    for challenger in candidates[1:]:
        votes_for_challenger = 0
        for judge_model in judge_models:
            response = await judge_model.judge(
                "Choose the better candidate. Reply A or B only.\n"
                f"A: {winner.final_output}\n"
                f"B: {challenger.final_output}"
            )
            label = response.raw_response.strip().split()[0].lower()
            if label == "b":
                votes_for_challenger += 1
        if votes_for_challenger > len(judge_models) / 2:
            winner = challenger
    return winner
