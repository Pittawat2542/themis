"""Public protocol contracts for engines, repositories, and exporters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.report import EvaluationReport
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.specs.base import SpecBase
from themis.specs.experiment import (
    DataItemContext,
    PromptMessage,
    PromptTemplateSpec,
    RuntimeContext,
    TrialSpec,
)
from themis.specs.foundational import JudgeInferenceSpec
from themis.types.events import ScoreRow, TrialEvent, TrialEventType, TrialSummaryRow
from themis.types.json_types import JSONValueType

if TYPE_CHECKING:
    from themis.runtime.timeline_view import RecordTimelineView

DatasetContext = DataItemContext | Mapping[str, object]
MetricContext = Mapping[str, object]


class InferenceResult(BaseModel):
    """Protocol envelope separating raw inference from optional conversation state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inference: InferenceRecord
    conversation: Conversation | None = None


class RenderedPrompt(BaseModel):
    """Rendered prompt envelope passed through inference hooks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[PromptMessage] = Field(default_factory=list)


@runtime_checkable
class InferenceEngine(Protocol):
    """Inference backend responsible for producing one candidate result."""

    def infer(
        self,
        trial: TrialSpec,
        context: DatasetContext,
        runtime: RuntimeContext,
    ) -> InferenceResult:
        """Return the raw inference output for one planned trial."""
        ...


@runtime_checkable
class Extractor(Protocol):
    """Structured parser that turns raw model output into extracted data."""

    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        """Parse one candidate into a structured extraction record."""
        ...


@runtime_checkable
class Metric(Protocol):
    """Scorer that turns a candidate plus context into one `MetricScore`."""

    def score(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        context: MetricContext,
    ) -> MetricScore:
        """Score one candidate against the provided metric context."""
        ...


@runtime_checkable
class JudgeService(Protocol):
    """Service object used by metrics that need extra judge-model calls."""

    def judge(
        self,
        metric_id: str,
        parent_candidate: CandidateRecord,
        judge_spec: JudgeInferenceSpec,
        prompt: PromptTemplateSpec,
        runtime: MetricContext,
    ) -> InferenceRecord:
        """Run one judge-model call and return the resulting inference record."""
        ...

    def consume_audit_trail(self, candidate_hash: str) -> JudgeAuditTrail | None:
        """Return and clear any judge audit trail recorded for a candidate."""
        ...


@runtime_checkable
class CandidateSelectionStrategy(Protocol):
    """Selects one best candidate from a materialized candidate set."""

    def select(self, candidates: Sequence[CandidateRecord]) -> CandidateRecord | None:
        """Choose the best candidate from a completed candidate collection."""
        ...


class PipelineHook(Protocol):
    """Pure transforms around inference, extraction, and evaluation stages."""

    def pre_inference(self, trial: TrialSpec, prompt: RenderedPrompt) -> RenderedPrompt:
        """Adjust the rendered prompt before inference runs."""
        ...

    def post_inference(
        self, trial: TrialSpec, result: InferenceResult
    ) -> InferenceResult:
        """Adjust the inference result before extraction begins."""
        ...

    def pre_extraction(
        self, trial: TrialSpec, candidate: CandidateRecord
    ) -> CandidateRecord:
        """Adjust a candidate before extractor execution."""
        ...

    def post_extraction(
        self, trial: TrialSpec, candidate: CandidateRecord
    ) -> CandidateRecord:
        """Adjust a candidate after extraction completes."""
        ...

    def pre_eval(self, trial: TrialSpec, candidate: CandidateRecord) -> CandidateRecord:
        """Adjust a candidate before metric scoring."""
        ...

    def post_eval(
        self, trial: TrialSpec, candidate: CandidateRecord
    ) -> CandidateRecord:
        """Adjust a candidate after metric scoring."""
        ...


@runtime_checkable
class TrialEventRepository(Protocol):
    """
    Append-only write-side repository for typed trial lifecycle events.
    """

    def save_spec(self, spec: SpecBase) -> None:
        """Persist a canonical spec before related events are appended."""
        ...

    def append_event(self, event: TrialEvent) -> None:
        """Append one lifecycle event to the event log."""
        ...

    def last_event_index(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> int | None:
        """Return the highest persisted event sequence for a trial or candidate."""
        ...

    def get_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        """Load persisted events for a trial or one candidate within that trial."""
        ...

    def has_projection_for_revision(self, trial_hash: str, eval_revision: str) -> bool:
        """Return whether a projection event exists for the given revision."""
        ...

    def latest_terminal_event_type(self, trial_hash: str) -> TrialEventType | None:
        """Return the most recent terminal event type for a trial, if any."""
        ...


@runtime_checkable
class ProjectionHandler(Protocol):
    """
    Trial-completion hook that emits or refreshes materialized projections.
    """

    def on_trial_completed(
        self, trial_hash: str, eval_revision: str = "latest"
    ) -> TrialRecord | None:
        """Refresh projections for a completed trial and return the materialized record."""
        ...


@runtime_checkable
class ProjectionRepository(Protocol):
    """
    Read-side projection/materialization contract over persisted trial events.
    """

    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        """Load a materialized trial record for one revision."""
        ...

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        """Load the recorded conversation for one candidate, if present."""
        ...

    def get_record_timeline(
        self,
        record_id: str,
        record_type: Literal["trial", "candidate"],
        eval_revision: str,
    ) -> RecordTimeline | None:
        """Load the stored timeline model for one trial or candidate."""
        ...

    def get_timeline_view(
        self,
        record_id: str,
        record_type: Literal["trial", "candidate"],
        eval_revision: str,
    ) -> RecordTimelineView | None:
        """Load the rich timeline view used by operators and diagnostics."""
        ...

    def materialize_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord:
        """Replay trial events into a materialized trial record."""
        ...

    def iter_candidate_scores(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        eval_revision: str = "latest",
    ) -> Iterator[ScoreRow]:
        """Iterate flattened metric score rows for reporting and comparisons."""
        ...

    def iter_trial_summaries(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
    ) -> Iterator[TrialSummaryRow]:
        """Iterate trial summary rows for reporting and comparison joins."""
        ...

    def save_trial_record(
        self, record: TrialRecord, *, eval_revision: str = "latest"
    ) -> None:
        """Persist a fully materialized trial record into projection tables."""
        ...

    def has_trial(self, trial_hash: str, eval_revision: str = "latest") -> bool:
        """Return whether a completed materialized trial exists for a revision."""
        ...


@runtime_checkable
class ReportExporter(Protocol):
    """Writer that serializes an assembled report to a target format."""

    def export(self, report: EvaluationReport, path: str) -> None:
        """Write one evaluation report to the target output path."""
        ...


EventRepository = TrialEventRepository
