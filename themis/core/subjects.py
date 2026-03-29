"""Evaluation subject models and validators."""

from __future__ import annotations

from pydantic import Field

from themis.core.base import HashableModel
from themis.core.models import ConversationTrace, GenerationResult, WorkflowTrace


class CandidateSetSubject(HashableModel):
    candidates: list[GenerationResult] = Field(default_factory=list, min_length=1)

    @property
    def size(self) -> int:
        return len(self.candidates)


class ConversationSubject(HashableModel):
    conversation: ConversationTrace


class TraceSubject(HashableModel):
    trace: WorkflowTrace


def validate_candidate_set_for_llm_metric(subject: CandidateSetSubject) -> None:
    if subject.size != 1:
        raise ValueError("LLM metrics require exactly one candidate.")


def validate_candidate_set_for_selection_metric(subject: CandidateSetSubject) -> None:
    if subject.size < 2:
        raise ValueError("Selection metrics require at least two candidates.")
