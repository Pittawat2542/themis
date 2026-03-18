"""Overlay-stage execution extracted from `TrialRunner`."""

from __future__ import annotations

import time
from collections.abc import Callable

from themis.errors.mapping import map_exception_to_error_record
from themis.orchestration.candidate_pipeline import (
    evaluate_candidate,
    transform_candidate,
)
from themis.orchestration.runner_events import TrialEventEmitter
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
)
from themis.records.candidate import CandidateRecord
from themis.records.judge import JudgeAuditTrail
from themis.registry.plugin_registry import PluginRegistry
from themis.storage.artifact_store import ArtifactStore
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    CandidateFailureEventMetadata,
    TimelineStage,
    TrialRetryEventMetadata,
    TrialEventType,
)
from themis.types.json_validation import dump_storage_json_bytes


class OverlayStageExecutor:
    """Owns output-transform and evaluation overlay execution."""

    def __init__(
        self,
        *,
        registry: PluginRegistry,
        event_emitter: TrialEventEmitter,
        artifact_store: ArtifactStore | None = None,
        max_retries: int,
        retry_backoff_factor: float,
        retryable_error_codes: tuple[ErrorCode, ...],
        telemetry_bus: TelemetryBus | None = None,
        append_session_event: Callable[..., None],
    ) -> None:
        self.registry = registry
        self.event_emitter = event_emitter
        self.artifact_store = artifact_store
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retryable_error_codes = retryable_error_codes
        self.telemetry_bus = telemetry_bus
        self.append_session_event = append_session_event

    def run_output_transform(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        transform: ResolvedOutputTransform,
    ) -> CandidateRecord:
        """Apply one output transform and append overlay-specific extraction events."""
        resolved_plugins = session.require_resolved_plugins()
        attempt = 0
        last_attempt_candidate: CandidateRecord | None = None
        while True:
            attempt += 1
            try:
                transformed_candidate = transform_candidate(
                    session.trial,
                    self.registry,
                    candidate,
                    transform,
                    telemetry_bus=self.telemetry_bus,
                    resolved_transform=resolved_plugins.output_transform_for(
                        transform.transform_hash
                    ),
                ).model_copy(update={"provenance": session.provenance})
                last_attempt_candidate = transformed_candidate
                self.event_emitter.emit_output_transform_events(
                    candidate.spec_hash,
                    transform,
                    transformed_candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                )
                if transformed_candidate.status != RecordStatus.ERROR:
                    return transformed_candidate
                self.event_emitter.emit_candidate_failure_event(
                    transformed_candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                    metadata=CandidateFailureEventMetadata(
                        transform_hash=transform.transform_hash,
                    ),
                )
                if not self._should_retry(transformed_candidate):
                    return transformed_candidate
            except Exception as exc:
                error = map_exception_to_error_record(
                    exc,
                    provider=session.trial.model.provider,
                    model_id=session.trial.model.model_id,
                    candidate_id=candidate.spec_hash,
                    attempt=attempt,
                )
                failed_candidate = candidate.model_copy(
                    update={
                        "status": RecordStatus.ERROR,
                        "error": error,
                        "provenance": session.provenance,
                    }
                )
                last_attempt_candidate = failed_candidate
                self.event_emitter.emit_candidate_failure_event(
                    failed_candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                    metadata=CandidateFailureEventMetadata(
                        transform_hash=transform.transform_hash,
                    ),
                )
                if not self._should_retry(failed_candidate):
                    return failed_candidate
            if attempt >= self.max_retries:
                return (
                    last_attempt_candidate
                    if last_attempt_candidate is not None
                    else candidate
                )
            self._append_retry_event(
                session,
                candidate_id=candidate.spec_hash,
                attempt=attempt,
                cand_index=candidate.sample_index,
                stage=TimelineStage.EXTRACTION,
                payload={
                    "attempt": attempt,
                    "cand_index": candidate.sample_index,
                    "transform_hash": transform.transform_hash,
                },
            )
            time.sleep(self._retry_delay_seconds(attempt))

    def run_evaluation_candidate(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        evaluation: ResolvedEvaluation,
    ) -> CandidateRecord:
        """Apply one evaluation and append overlay-specific scoring events."""
        resolved_plugins = session.require_resolved_plugins()
        attempt = 0
        last_attempt_candidate: CandidateRecord | None = None
        while True:
            attempt += 1
            judge_service = resolved_plugins.create_judge_service()
            try:
                evaluated_candidate = evaluate_candidate(
                    session.trial,
                    self.registry,
                    session.dataset_context,
                    session.base_runtime,
                    candidate,
                    evaluation,
                    judge_service=judge_service,
                    telemetry_bus=self.telemetry_bus,
                    resolved_evaluation=resolved_plugins.evaluation_for(
                        evaluation.evaluation_hash
                    ),
                )
                judge_trails = self._collect_judge_trails(
                    judge_service, candidate.spec_hash
                )
                judge_artifact_refs = self._persist_judge_artifacts(judge_trails)
                evaluated_candidate = self._apply_judge_artifacts(
                    evaluated_candidate,
                    judge_artifact_refs,
                ).model_copy(update={"provenance": session.provenance})
                last_attempt_candidate = evaluated_candidate
                self.event_emitter.emit_evaluation_candidate_events(
                    candidate.spec_hash,
                    evaluation,
                    evaluated_candidate,
                    judge_call_count=sum(
                        len(trail.judge_calls) for trail in judge_trails
                    ),
                    judge_artifact_refs=judge_artifact_refs,
                    append_event=lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                )
                if evaluated_candidate.status != RecordStatus.ERROR:
                    return evaluated_candidate
                self.event_emitter.emit_candidate_failure_event(
                    evaluated_candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                    metadata=CandidateFailureEventMetadata(
                        transform_hash=(
                            evaluation.transform.transform_hash
                            if evaluation.transform is not None
                            else None
                        ),
                        evaluation_hash=evaluation.evaluation_hash,
                    ),
                )
                if not self._should_retry(evaluated_candidate):
                    return evaluated_candidate
            except Exception as exc:
                error = map_exception_to_error_record(
                    exc,
                    provider=session.trial.model.provider,
                    model_id=session.trial.model.model_id,
                    candidate_id=candidate.spec_hash,
                    attempt=attempt,
                )
                failed_candidate = candidate.model_copy(
                    update={
                        "status": RecordStatus.ERROR,
                        "error": error,
                        "provenance": session.provenance,
                    }
                )
                last_attempt_candidate = failed_candidate
                self.event_emitter.emit_candidate_failure_event(
                    failed_candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                    metadata=CandidateFailureEventMetadata(
                        transform_hash=(
                            evaluation.transform.transform_hash
                            if evaluation.transform is not None
                            else None
                        ),
                        evaluation_hash=evaluation.evaluation_hash,
                    ),
                )
                if not self._should_retry(failed_candidate):
                    return failed_candidate
            if attempt >= self.max_retries:
                return (
                    last_attempt_candidate
                    if last_attempt_candidate is not None
                    else candidate
                )
            self._append_retry_event(
                session,
                candidate_id=candidate.spec_hash,
                attempt=attempt,
                cand_index=candidate.sample_index,
                stage=TimelineStage.EVALUATION,
                payload={
                    "attempt": attempt,
                    "cand_index": candidate.sample_index,
                    "evaluation_hash": evaluation.evaluation_hash,
                    "transform_hash": (
                        evaluation.transform.transform_hash
                        if evaluation.transform is not None
                        else None
                    ),
                },
            )
            time.sleep(self._retry_delay_seconds(attempt))

    def _collect_judge_trails(
        self,
        judge_service,
        candidate_id: str,
    ) -> list[JudgeAuditTrail]:
        trail = judge_service.consume_audit_trail(candidate_id)
        return [trail] if trail is not None else []

    def _persist_judge_artifacts(
        self, judge_trails: list[JudgeAuditTrail]
    ) -> list[ArtifactRef]:
        if self.artifact_store is None:
            return []

        artifact_refs: list[ArtifactRef] = []
        for trail in judge_trails:
            artifact_hash = self.artifact_store.put_blob(
                dump_storage_json_bytes(
                    trail.model_dump(mode="json"), label="judge_audit"
                ),
                "application/json",
            )
            artifact_refs.append(
                ArtifactRef(
                    artifact_hash=artifact_hash,
                    media_type="application/json",
                    label=trail.spec_hash,
                    role=ArtifactRole.JUDGE_AUDIT,
                )
            )
        return artifact_refs

    def _apply_judge_artifacts(
        self,
        candidate: CandidateRecord,
        artifact_refs: list[ArtifactRef],
    ) -> CandidateRecord:
        if not artifact_refs:
            return candidate
        return candidate.model_copy(
            update={
                "judge_audits": [artifact.artifact_hash for artifact in artifact_refs]
            }
        )

    def _append_retry_event(
        self,
        session: TrialExecutionSession,
        *,
        candidate_id: str,
        attempt: int,
        cand_index: int,
        stage: TimelineStage,
        payload: dict[str, object],
    ) -> None:
        self.append_session_event(
            session,
            TrialEventType.TRIAL_RETRY,
            candidate_id=candidate_id,
            stage=stage,
            metadata=TrialRetryEventMetadata(
                attempt=attempt,
                cand_index=cand_index,
            ),
            payload=payload,
        )

    def _should_retry(self, candidate: CandidateRecord) -> bool:
        if candidate.status != RecordStatus.ERROR or candidate.error is None:
            return False
        if self.retryable_error_codes:
            return candidate.error.code in self.retryable_error_codes
        return bool(candidate.error.retryable)

    def _retry_delay_seconds(self, attempt: int) -> float:
        return min(5.0, 0.05 * (self.retry_backoff_factor ** max(attempt - 1, 0)))
