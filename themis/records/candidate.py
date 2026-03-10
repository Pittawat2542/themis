from __future__ import annotations

from pydantic import Field

from themis.records.base import RecordBase
from themis.records.conversation import Conversation
from themis.records.evaluation import EvaluationRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.timeline import RecordTimeline


class CandidateRecord(RecordBase):
    """Groups all artifacts produced for one candidate sample through the pipeline."""

    candidate_id: str | None = Field(
        default=None,
        description="Stable candidate identifier. Defaults to spec_hash when omitted.",
    )

    sample_index: int = Field(
        default=0,
        description="Zero-based sample index for deterministic candidate ordering.",
    )

    conversation: Conversation | None = Field(
        default=None,
        description="Candidate-scoped conversation trace, when the inference engine emits one.",
    )

    timeline: RecordTimeline | None = Field(
        default=None,
        description="Candidate-scoped stage timeline projected from append-only lifecycle events.",
    )

    inference: InferenceRecord | None = None

    extractions: list[ExtractionRecord] = Field(default_factory=list)

    evaluation: EvaluationRecord | None = None

    judge_audits: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.candidate_id is None:
            object.__setattr__(self, "candidate_id", self.spec_hash)

    def best_extraction(self) -> ExtractionRecord | None:
        for extraction in self.extractions:
            if extraction.success:
                return extraction
        return self.extractions[0] if self.extractions else None
