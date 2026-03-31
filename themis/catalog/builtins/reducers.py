"""Builtin reducers."""

from __future__ import annotations

from collections import Counter

from themis.core.contexts import ReduceContext
from themis.core.models import GenerationResult, ReducedCandidate


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
