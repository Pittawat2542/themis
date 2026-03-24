"""Typed lifecycle events and metadata persisted during experiment execution."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from themis.records.error import ErrorRecord
from themis.types.enums import RecordStatus
from themis.types.json_types import JSONDict, JSONValueType


class TimelineStage(str, Enum):
    """Fine-grained timeline stage represented by a trial event."""

    ITEM_LOAD = "item_load"
    PROMPT_RENDER = "prompt_render"
    INFERENCE = "inference"
    EXTRACTION = "extraction"
    EVALUATION = "evaluation"
    PROJECTION = "projection"


class ArtifactRole(str, Enum):
    """Semantic role of an artifact linked from an event."""

    ITEM_PAYLOAD = "item_payload"
    RENDERED_PROMPT = "rendered_prompt"
    INFERENCE_OUTPUT = "inference_output"
    EXTRACTION_OUTPUT = "extraction_output"
    EVALUATION_OUTPUT = "evaluation_output"
    METRIC_DETAILS = "metric_details"
    JUDGE_AUDIT = "judge_audit"


class TrialEventType(str, Enum):
    """Discriminator for append-only trial events."""

    TRIAL_STARTED = "trial_started"
    ITEM_LOADED = "item_loaded"
    PROMPT_RENDERED = "prompt_rendered"
    CANDIDATE_STARTED = "candidate_started"
    CONVERSATION_EVENT = "conversation_event"
    INFERENCE_COMPLETED = "inference_completed"
    EXTRACTION_COMPLETED = "extraction_completed"
    EVALUATION_COMPLETED = "evaluation_completed"
    CANDIDATE_COMPLETED = "candidate_completed"
    CANDIDATE_FAILED = "candidate_failed"
    TRIAL_RETRY = "trial_retry"
    PROJECTION_COMPLETED = "projection_completed"
    TRIAL_COMPLETED = "trial_completed"
    TRIAL_FAILED = "trial_failed"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TrialEventMetadata(BaseModel):
    """Base type for structured event metadata persisted as JSON dicts."""

    model_config = ConfigDict(frozen=True, extra="allow")

    def as_dict(self) -> JSONDict:
        """Returns the metadata as a JSON-safe dictionary."""

        return cast(JSONDict, self.model_dump(mode="json"))


class EmptyEventMetadata(TrialEventMetadata):
    """Metadata shape for events that do not define structured metadata."""


class OverlayEventMetadata(TrialEventMetadata):
    """Metadata shared by transform/evaluation/projection scoped events."""

    transform_hash: str | None = None
    evaluation_hash: str | None = None


class ItemLoadedEventMetadata(TrialEventMetadata):
    """Metadata persisted when a dataset item is loaded."""

    item_id: str | None = None
    dataset_source: str | None = None
    dataset_revision: str | None = None
    tags: JSONDict | None = None
    item_payload_hash: str | None = None


class PromptRenderedEventMetadata(TrialEventMetadata):
    """Metadata recorded after prompt rendering succeeds."""

    prompt_template_id: str | None = None
    rendered_prompt_hash: str | None = None
    input_field_map: list[str] = Field(default_factory=list)


class InferenceCompletedEventMetadata(TrialEventMetadata):
    """Provider and token-usage details captured after inference."""

    provider: str | None = None
    model_id: str | None = None
    inference_params_hash: str | None = None
    effective_seed: int | None = None
    provider_request_id: str | None = None
    token_usage: JSONDict = Field(default_factory=dict)


class ExtractionCompletedEventMetadata(OverlayEventMetadata):
    """Metadata for extractor execution scoped to a transform overlay."""

    extractor_id: str | None = None
    attempt_index: int | None = None
    success: bool | None = None
    failure_reason: str | None = None


class EvaluationCompletedEventMetadata(OverlayEventMetadata):
    """Metadata for metric evaluation scoped to an evaluation overlay."""

    metric_id: str | None = None
    score: float | None = None
    judge_call_count: int | None = None
    details_hash: str | None = None
    judge_audit_hashes: list[str] = Field(default_factory=list)


class CandidateFailureEventMetadata(OverlayEventMetadata):
    """Metadata carried by failed transform/evaluation candidate events."""


class ProjectionCompletedEventMetadata(OverlayEventMetadata):
    """Metadata for projection artifacts derived from prior events."""

    projection_version: str | None = None
    source_event_range: list[int] | None = None


class TrialRetryEventMetadata(TrialEventMetadata):
    """Retry metadata identifying which candidate attempt was re-run."""

    attempt: int | None = None
    cand_index: int | None = None


class ArtifactRef(BaseModel):
    """Reference to a deduplicated artifact relevant to an event or timeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_hash: str
    media_type: str
    label: str | None = None
    role: ArtifactRole | None = None


class TrialEvent(BaseModel):
    """Typed append-only lifecycle event persisted by the trial event repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    event_seq: int
    event_id: str
    event_type: TrialEventType
    candidate_id: str | None = None
    stage: TimelineStage | None = None
    status: RecordStatus | None = None
    event_ts: datetime = Field(default_factory=_now_utc)
    metadata: TrialEventMetadata = Field(default_factory=EmptyEventMetadata)
    payload: JSONValueType | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: ErrorRecord | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_typed_metadata(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        event_type = data.get("event_type")
        if event_type is None:
            return data
        normalized = dict(data)
        normalized["metadata"] = parse_trial_event_metadata(
            event_type=event_type,
            metadata=normalized.get("metadata"),
        )
        return normalized


class ScoreRow(BaseModel):
    """Flattened candidate-level metric projection row used by stats/reporting code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    candidate_id: str
    metric_id: str
    score: float
    details: JSONDict = Field(default_factory=dict)


class TraceScoreRow(BaseModel):
    """Flattened trace-level metric projection row used by trace analysis code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    trace_id: str
    trace_scope: str
    trace_score_hash: str
    metric_id: str
    score: float
    details: JSONDict = Field(default_factory=dict)


class TrialSummaryRow(BaseModel):
    """Flattened trial-level summary row used by reporting and comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_hash: str
    model_id: str | None = None
    task_id: str | None = None
    benchmark_id: str | None = None
    slice_id: str | None = None
    prompt_variant_id: str | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)
    item_id: str | None = None
    status: RecordStatus


def metadata_model_for_event(
    event_type: TrialEventType | str,
) -> type[TrialEventMetadata]:
    """Returns the metadata model used for a given event discriminator."""

    resolved_event_type = TrialEventType(event_type)
    if resolved_event_type == TrialEventType.ITEM_LOADED:
        return ItemLoadedEventMetadata
    if resolved_event_type == TrialEventType.PROMPT_RENDERED:
        return PromptRenderedEventMetadata
    if resolved_event_type == TrialEventType.INFERENCE_COMPLETED:
        return InferenceCompletedEventMetadata
    if resolved_event_type == TrialEventType.EXTRACTION_COMPLETED:
        return ExtractionCompletedEventMetadata
    if resolved_event_type == TrialEventType.EVALUATION_COMPLETED:
        return EvaluationCompletedEventMetadata
    if resolved_event_type == TrialEventType.CANDIDATE_FAILED:
        return CandidateFailureEventMetadata
    if resolved_event_type == TrialEventType.PROJECTION_COMPLETED:
        return ProjectionCompletedEventMetadata
    if resolved_event_type == TrialEventType.TRIAL_RETRY:
        return TrialRetryEventMetadata
    return EmptyEventMetadata


def parse_trial_event_metadata(
    *,
    event_type: TrialEventType | str,
    metadata: TrialEventMetadata | JSONDict | None,
) -> TrialEventMetadata:
    """Coerces persisted metadata into the typed model for ``event_type``."""

    model_type = metadata_model_for_event(event_type)
    if isinstance(metadata, model_type):
        return metadata
    payload = (
        metadata.as_dict() if isinstance(metadata, TrialEventMetadata) else metadata
    )
    return model_type.model_validate(payload or {})
