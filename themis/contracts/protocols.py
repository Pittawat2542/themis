"""Public protocol contracts for engines, repositories, and exporters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.observability import ObservabilityLink, ObservabilitySnapshot
from themis.records.report import EvaluationReport
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.specs.base import SpecBase
from themis.specs.experiment import (
    DataItemContext,
    PromptMessage,
    PromptTurnSpec,
    PromptTemplateSpec,
    RuntimeContext,
    TrialSpec,
)
from themis.specs.foundational import (
    JudgeInferenceSpec,
    McpServerSpec,
    TaskSpec,
    ToolSpec,
)
from themis.types.enums import RecordType
from themis.types.events import (
    ScoreRow,
    TraceScoreRow,
    TrialEvent,
    TrialEventType,
    TrialSummaryRow,
)
from themis.types.json_types import JSONValueType

if TYPE_CHECKING:
    from themis.benchmark.query import DatasetQuerySpec
    from themis.benchmark.specs import DatasetSliceSpec
    from themis.runtime.trace_view import TraceView
    from themis.runtime.timeline_view import RecordTimelineView

DatasetContext = DataItemContext | Mapping[str, object]
DatasetItem = DataItemContext | Mapping[str, object] | JSONValueType
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
    follow_up_turns: list[PromptTurnSpec] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)


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
class TrialMetric(Protocol):
    """Scorer that turns a candidate set plus context into one `MetricScore`."""

    def score_trial(
        self,
        trial: TrialSpec,
        candidates: Sequence[CandidateRecord],
        context: MetricContext,
    ) -> MetricScore:
        """Score one full candidate set against the provided metric context."""
        ...


@runtime_checkable
class TraceMetric(Protocol):
    """Scorer that turns one persisted execution trace into one `MetricScore`."""

    def score_trace(
        self,
        trace: TraceView,
        context: MetricContext,
    ) -> MetricScore:
        """Score one candidate or trial trace against the provided metric context."""
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
class DatasetLoader(Protocol):
    """Loader that materializes dataset items for one task."""

    def load_task_items(self, task: TaskSpec) -> Sequence[DatasetItem]:
        """Return execution items for the supplied task-like object."""
        ...


@runtime_checkable
class DatasetProvider(Protocol):
    """Query-aware dataset provider used by the benchmark-first public surface."""

    def scan(
        self,
        slice_spec: DatasetSliceSpec,
        query: DatasetQuerySpec,
    ) -> Sequence[DatasetItem]:
        """Return slice items after applying the requested dataset query."""
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

    def has_projection_for_overlay(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        """Return whether a projection event exists for the requested overlay."""
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
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        """Refresh projections for a completed trial overlay and return the record."""
        ...


@runtime_checkable
class ProjectionRepository(Protocol):
    """
    Read-side projection/materialization contract over persisted trial events.
    """

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        """Load a materialized trial record for one overlay."""
        ...

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        """Load the recorded conversation for one candidate, if present."""
        ...

    def get_record_timeline(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimeline | None:
        """Load the stored timeline model for one trial or candidate."""
        ...

    def get_timeline_view(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        """Load the rich timeline view used by operators and diagnostics."""
        ...

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
    ) -> TrialRecord:
        """Replay trial events into a materialized trial record."""
        ...

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[ScoreRow]:
        """Iterate flattened metric score rows for reporting and comparisons."""
        ...

    def iter_trial_summaries(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[TrialSummaryRow]:
        """Iterate trial summary rows for reporting and comparison joins."""
        ...

    def iter_trace_scores(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        metric_id: str | None = None,
        trace_score_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[TraceScoreRow]:
        """Iterate flattened trace score rows for reporting and comparisons."""
        ...

    def save_trial_record(
        self,
        record: TrialRecord,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> None:
        """Persist a fully materialized trial record into projection tables."""
        ...

    def has_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        """Return whether a completed materialized trial exists for an overlay."""
        ...


@runtime_checkable
class ProjectionRefreshRepository(Protocol):
    """Minimal materialization contract used by projection refresh orchestration."""

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
    ) -> TrialRecord:
        """Replay trial events into a refreshed trial record."""
        ...


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed blob persistence used for large payloads and audits."""

    def put_blob(self, blob: bytes, media_type: str) -> str:
        """Persist one raw blob and return its stable reference."""
        ...

    def get_blob(self, ref: str) -> bytes:
        """Load one blob by reference."""
        ...

    def write_json(self, data: JSONValueType) -> tuple[str, str]:
        """Persist one JSON payload and return its URI plus blob hash."""
        ...

    def read_json(self, sha256_hash: str) -> JSONValueType:
        """Load one JSON payload by its blob hash reference."""
        ...

    def exists(self, sha256_hash: str) -> bool:
        """Return whether one blob exists."""
        ...


@runtime_checkable
class ObservabilityStore(Protocol):
    """Persistence for provider-neutral observability links."""

    def save_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        snapshot: ObservabilitySnapshot,
    ) -> None:
        """Persist or replace one observability snapshot."""
        ...

    def save_link(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        link: ObservabilityLink,
    ) -> None:
        """Persist or replace one provider-specific observability link."""
        ...

    def get_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
    ) -> ObservabilitySnapshot | None:
        """Load the persisted observability snapshot for one record overlay."""
        ...


@runtime_checkable
class ReportExporter(Protocol):
    """Writer that serializes an assembled report to a target format."""

    def export(self, report: EvaluationReport, path: str) -> None:
        """Write one evaluation report to the target output path."""
        ...


EventRepository = TrialEventRepository
