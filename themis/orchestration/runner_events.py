"""Internal event emission helpers for the staged trial runner."""

from __future__ import annotations

from collections.abc import Callable

from themis._replay import ResumeState
from themis.contracts.protocols import TrialEventRepository
from themis.orchestration.runner_state import (
    CandidateStageResults,
    TrialExecutionSession,
    artifact_ref,
)
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
)
from themis.records.candidate import CandidateRecord
from themis.records.error import ErrorRecord
from themis.specs.experiment import TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.telemetry.bus import TelemetryBus, TelemetryEventName
from themis.types.enums import ErrorWhere, RecordStatus
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    EmptyEventMetadata,
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    InferenceCompletedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventMetadata,
    TrialEventType,
)
from themis.types.json_types import JSONValueType

AppendEvent = Callable[..., None]


class TrialEventEmitter:
    """Owns trial-event persistence plus the higher-level stage event helpers."""

    def __init__(
        self,
        event_repo: TrialEventRepository,
        *,
        artifact_store: ArtifactStore | None = None,
        telemetry_bus: TelemetryBus | None = None,
    ) -> None:
        self.event_repo = event_repo
        self.artifact_store = artifact_store
        self.telemetry_bus = telemetry_bus

    def append_session_event(
        self,
        session: TrialExecutionSession,
        event_type: TrialEventType,
        *,
        candidate_id: str | None = None,
        stage: TimelineStage | None = None,
        status: RecordStatus | None = None,
        metadata: TrialEventMetadata | None = None,
        payload: JSONValueType | None = None,
        artifact_refs: list[ArtifactRef] | None = None,
        error: ErrorRecord | None = None,
    ) -> None:
        """Append one event to the shared session stream and mirror telemetry."""
        with session.event_lock:
            session.event_seq += 1
            event = TrialEvent(
                trial_hash=session.trial_hash,
                event_seq=session.event_seq,
                event_id=f"{session.trial_hash}:{session.event_seq}",
                candidate_id=candidate_id,
                event_type=event_type,
                stage=stage,
                status=status,
                metadata=metadata or EmptyEventMetadata(),
                payload=payload,
                artifact_refs=artifact_refs or [],
                error=error,
            )
            self.event_repo.append_event(event)
            self._emit_telemetry_event(event)

    def emit_candidate_stage_events(
        self,
        trial: TrialSpec,
        stage_results: CandidateStageResults,
        append_event: AppendEvent,
        *,
        resume_state: ResumeState | None = None,
        judge_call_count: int = 0,
        judge_artifact_refs: list[ArtifactRef] | None = None,
    ) -> None:
        """Persist the generated, transformed, and evaluated candidate stage events."""
        candidate = stage_results.generated_candidate
        candidate_id = candidate.spec_hash
        resume_index = (
            resume_state.last_event_index if resume_state is not None else None
        )
        self._emit_inference_events(
            trial,
            candidate,
            append_event,
            resume_index=resume_index,
        )
        for transform, transformed_candidate in stage_results.transformed_candidates:
            self.emit_output_transform_events(
                candidate_id,
                transform,
                transformed_candidate,
                append_event,
            )
        for evaluation, evaluated_candidate in stage_results.evaluated_candidates:
            self.emit_evaluation_candidate_events(
                candidate_id,
                evaluation,
                evaluated_candidate,
                judge_call_count=judge_call_count,
                judge_artifact_refs=judge_artifact_refs,
                append_event=append_event,
            )

    def emit_candidate_failure_event(
        self,
        candidate: CandidateRecord,
        append_event: AppendEvent,
        *,
        metadata: TrialEventMetadata | None = None,
    ) -> None:
        """Persist one candidate failure event when a stage returns an error candidate."""
        if candidate.status != RecordStatus.ERROR or candidate.error is None:
            return
        append_event(
            TrialEventType.CANDIDATE_FAILED,
            candidate_id=candidate.spec_hash,
            stage=self._stage_for_error(candidate.error.where),
            status=RecordStatus.ERROR,
            metadata=metadata,
            error=candidate.error,
        )

    def emit_output_transform_events(
        self,
        candidate_id: str,
        transform: ResolvedOutputTransform,
        transformed_candidate: CandidateRecord,
        append_event: AppendEvent,
    ) -> None:
        """Persist extraction events for one output-transform overlay."""
        for attempt_index, extraction in enumerate(transformed_candidate.extractions):
            extraction_payload = extraction.model_dump(mode="json")
            extraction_artifact = artifact_ref(
                extraction_payload,
                role=ArtifactRole.EXTRACTION_OUTPUT,
                label="extraction_output",
                artifact_store=self.artifact_store,
            )
            append_event(
                TrialEventType.EXTRACTION_COMPLETED,
                candidate_id=candidate_id,
                stage=TimelineStage.EXTRACTION,
                status=RecordStatus.OK if extraction.success else RecordStatus.ERROR,
                metadata=ExtractionCompletedEventMetadata(
                    transform_hash=transform.transform_hash,
                    extractor_id=extraction.extractor_id,
                    attempt_index=attempt_index,
                    success=extraction.success,
                    failure_reason=extraction.failure_reason,
                ),
                payload=extraction_payload,
                artifact_refs=[extraction_artifact[0]],
            )

    def emit_evaluation_candidate_events(
        self,
        candidate_id: str,
        evaluation: ResolvedEvaluation,
        evaluated_candidate: CandidateRecord,
        *,
        judge_call_count: int,
        judge_artifact_refs: list[ArtifactRef] | None,
        append_event: AppendEvent,
    ) -> None:
        """Persist metric-scoring events for one evaluation overlay."""
        if (
            evaluated_candidate.evaluation is None
            or not evaluated_candidate.evaluation.metric_scores
        ):
            return
        first_score = evaluated_candidate.evaluation.metric_scores[0]
        details_artifact = (
            artifact_ref(
                first_score.details,
                role=ArtifactRole.METRIC_DETAILS,
                label="metric_details",
                artifact_store=self.artifact_store,
            )
            if first_score.details
            else None
        )
        evaluation_payload = evaluated_candidate.evaluation.model_dump(mode="json")
        evaluation_artifact = artifact_ref(
            evaluation_payload,
            role=ArtifactRole.EVALUATION_OUTPUT,
            label="evaluation_output",
            artifact_store=self.artifact_store,
        )
        append_event(
            TrialEventType.EVALUATION_COMPLETED,
            candidate_id=candidate_id,
            stage=TimelineStage.EVALUATION,
            status=RecordStatus.OK,
            metadata=EvaluationCompletedEventMetadata(
                transform_hash=(
                    evaluation.transform.transform_hash
                    if evaluation.transform is not None
                    else None
                ),
                evaluation_hash=evaluation.evaluation_hash,
                metric_id=first_score.metric_id,
                score=first_score.value,
                judge_call_count=judge_call_count,
                details_hash=details_artifact[1]
                if details_artifact is not None
                else None,
                judge_audit_hashes=[
                    artifact.artifact_hash for artifact in judge_artifact_refs or []
                ],
            ),
            payload=evaluation_payload,
            artifact_refs=[
                *([details_artifact[0]] if details_artifact is not None else []),
                evaluation_artifact[0],
                *(judge_artifact_refs or []),
            ],
        )

    def _emit_inference_events(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        append_event: AppendEvent,
        *,
        resume_index: int | None,
    ) -> None:
        if candidate.inference is None:
            return
        token_usage = candidate.inference.token_usage
        inference_payload = candidate.inference.model_dump(mode="json")
        inference_artifact = artifact_ref(
            inference_payload,
            role=ArtifactRole.INFERENCE_OUTPUT,
            label="inference_output",
            artifact_store=self.artifact_store,
        )
        append_event(
            TrialEventType.INFERENCE_COMPLETED,
            candidate_id=candidate.spec_hash,
            stage=TimelineStage.INFERENCE,
            status=RecordStatus.OK,
            metadata=InferenceCompletedEventMetadata(
                provider=trial.model.provider,
                model_id=trial.model.model_id,
                inference_params_hash=trial.params.spec_hash,
                provider_request_id=candidate.inference.provider_request_id,
                token_usage=(
                    token_usage.model_dump(mode="json")
                    if token_usage is not None
                    else {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ),
            ),
            payload=inference_payload,
            artifact_refs=[inference_artifact[0]],
        )
        if candidate.conversation is None:
            return
        for conversation_event in candidate.conversation.events:
            if (
                resume_index is not None
                and conversation_event.event_index <= resume_index
            ):
                continue
            append_event(
                TrialEventType.CONVERSATION_EVENT,
                candidate_id=candidate.spec_hash,
                payload=conversation_event.model_dump(mode="json"),
            )

    def _stage_for_error(self, where: ErrorWhere) -> TimelineStage:
        if where == ErrorWhere.EXTRACTOR:
            return TimelineStage.EXTRACTION
        if where == ErrorWhere.METRIC:
            return TimelineStage.EVALUATION
        return TimelineStage.INFERENCE

    def _emit_telemetry_event(self, event: TrialEvent) -> None:
        if self.telemetry_bus is None:
            return
        payload: dict[str, JSONValueType] = {"trial_hash": event.trial_hash}
        if event.candidate_id is not None:
            payload["candidate_id"] = event.candidate_id
        if event.status is not None:
            payload["status"] = event.status.value
        if event.error is not None:
            payload["code"] = event.error.code.value
            payload["where"] = event.error.where.value
            payload["message"] = event.error.message
        if isinstance(event.payload, dict):
            payload.update(event.payload)

        if event.event_type == TrialEventType.TRIAL_STARTED:
            self.telemetry_bus.emit(TelemetryEventName.TRIAL_START, **payload)
            return
        if event.event_type in {
            TrialEventType.TRIAL_COMPLETED,
            TrialEventType.TRIAL_FAILED,
        }:
            self.telemetry_bus.emit(TelemetryEventName.TRIAL_END, **payload)
            if event.error is not None:
                self.telemetry_bus.emit(TelemetryEventName.ERROR, **payload)
            return
        if event.event_type == TrialEventType.CONVERSATION_EVENT:
            self.telemetry_bus.emit(TelemetryEventName.CONVERSATION_EVENT, **payload)
            kind = (
                event.payload.get("kind") if isinstance(event.payload, dict) else None
            )
            if kind == "tool_call":
                self.telemetry_bus.emit(TelemetryEventName.TOOL_CALL, **payload)
