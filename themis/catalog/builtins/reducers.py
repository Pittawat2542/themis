"""Builtin reducers."""

from __future__ import annotations

from collections import Counter

from themis.core.contexts import ReduceContext
from themis.core.models import GenerationResult, ReducedCandidate
from themis.core.protocols import JudgeModel


class MajorityVoteReducer:
    component_id = "builtin/majority_vote"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-majority-vote-fingerprint"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        serialized_outputs = [_stable_output(candidate.final_output) for candidate in candidates]
        winning_output, _count = Counter(serialized_outputs).most_common(1)[0]
        winning_candidate = next(
            candidate
            for candidate, serialized in zip(candidates, serialized_outputs, strict=False)
            if serialized == winning_output
        )
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=winning_candidate.final_output,
            metadata={"strategy": "majority_vote"},
        )


def _stable_output(value: object) -> str:
    return repr(value)


class BestOfNReducer:
    component_id = "builtin/best_of_n"
    version = "1.0"

    def fingerprint(self) -> str:
        return "builtin-best-of-n-fingerprint"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        if len(candidates) <= 1 or not ctx.judge_models:
            winner = candidates[0]
        else:
            winner = await _select_best_candidate(candidates, ctx.judge_models)
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=winner.final_output,
            metadata={"strategy": "best_of_n"},
        )


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
