"""Generation-stage candidate execution extracted from `TrialRunner`."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from themis._replay import ResumeState
from themis.errors.mapping import map_exception_to_error_record
from themis.orchestration.candidate_pipeline import (
    candidate_hash_for_index,
    generate_candidate,
)
from themis.orchestration.runner_events import TrialEventEmitter
from themis.orchestration.runner_state import (
    CandidateStageResults,
    TrialExecutionSession,
    candidate_from_terminal_events,
    resume_state_from_events,
)
from themis.records.candidate import CandidateRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import RuntimeContext
from themis.telemetry.bus import TelemetryBus, TelemetryEventName
from themis.types.enums import RecordStatus
from themis.types.events import (
    PromptRenderedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialRetryEventMetadata,
    TrialEventType,
)


class GenerationStageExecutor:
    """Owns generation-stage candidate execution and retry behavior."""

    def __init__(
        self,
        *,
        registry: PluginRegistry,
        event_emitter: TrialEventEmitter,
        max_retries: int,
        project_seed: int | None = None,
        telemetry_bus: TelemetryBus | None = None,
        append_session_event: Callable[..., None],
        get_trial_events: Callable[[str, str | None], list[TrialEvent]],
    ) -> None:
        self.registry = registry
        self.event_emitter = event_emitter
        self.max_retries = max_retries
        self.project_seed = project_seed
        self.telemetry_bus = telemetry_bus
        self.append_session_event = append_session_event
        self.get_trial_events = get_trial_events

    def run_candidate(
        self,
        session: TrialExecutionSession,
        cand_index: int,
    ) -> CandidateRecord:
        """Execute the generation stage for one candidate within a prepared trial."""
        resolved_plugins = session.require_resolved_plugins()
        trial = session.trial
        candidate_id = candidate_hash_for_index(trial, cand_index)
        candidate_events = self.get_trial_events(session.trial_hash, candidate_id)
        terminal_candidate = candidate_from_terminal_events(
            candidate_id,
            cand_index,
            candidate_events,
        )
        if terminal_candidate is not None:
            return terminal_candidate
        resume_state = resume_state_from_events(candidate_id, candidate_events)

        if not candidate_events:
            prompt_metadata = PromptRenderedEventMetadata(
                prompt_template_id=trial.prompt.id,
                rendered_prompt_hash=session.prompt_artifact[1],
                input_field_map=sorted(session.dataset_context.keys()),
            )
            self.append_session_event(
                session,
                TrialEventType.CANDIDATE_STARTED,
                candidate_id=candidate_id,
                payload={"sample_index": cand_index},
            )
            self.append_session_event(
                session,
                TrialEventType.PROMPT_RENDERED,
                candidate_id=candidate_id,
                stage=TimelineStage.PROMPT_RENDER,
                status=RecordStatus.OK,
                metadata=prompt_metadata,
                payload=session.prompt_payload,
                artifact_refs=[session.prompt_artifact[0]],
            )

        attempt = 0
        while True:
            attempt += 1
            stage_results: CandidateStageResults | None = None
            try:
                generated_candidate = generate_candidate(
                    trial,
                    self.registry,
                    session.dataset_context,
                    self._candidate_runtime_context(
                        session.base_runtime,
                        resume_state=resume_state,
                        candidate_seed=self._candidate_seed(
                            trial.spec_hash, cand_index
                        ),
                    ),
                    cand_index,
                    resolved_generation=resolved_plugins.generation,
                )
                stage_results = CandidateStageResults(
                    generated_candidate=generated_candidate,
                    transformed_candidates=(),
                    evaluated_candidates=(),
                    final_candidate=generated_candidate,
                )
            except Exception as exc:
                error = map_exception_to_error_record(
                    exc,
                    provider=trial.model.provider,
                    model_id=trial.model.model_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                )
                candidate = CandidateRecord(
                    spec_hash=candidate_id,
                    candidate_id=candidate_id,
                    sample_index=cand_index,
                    status=RecordStatus.ERROR,
                    error=error,
                    conversation=resume_state.conversation
                    if resume_state is not None
                    else None,
                    provenance=session.provenance,
                )
                if self.telemetry_bus is not None:
                    self.telemetry_bus.emit(
                        TelemetryEventName.ERROR,
                        trial_hash=session.trial_hash,
                        candidate_id=candidate_id,
                        provider=trial.model.provider,
                        model_id=trial.model.model_id,
                        code=error.code.value,
                        where=error.where.value,
                        message=error.message,
                    )

            if stage_results is not None:
                candidate = stage_results.final_candidate.model_copy(
                    update={"provenance": session.provenance}
                )
                self.event_emitter.emit_candidate_stage_events(
                    trial,
                    stage_results,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                    resume_state=resume_state,
                    judge_call_count=0,
                    judge_artifact_refs=[],
                )
            else:
                candidate = candidate.model_copy(
                    update={"provenance": session.provenance}
                )
                self.event_emitter.emit_candidate_failure_event(
                    candidate,
                    lambda event_type, **kwargs: self.append_session_event(
                        session,
                        event_type,
                        **kwargs,
                    ),
                )

            if not (
                candidate.status == RecordStatus.ERROR
                and candidate.error
                and candidate.error.retryable
            ):
                self._append_candidate_completed(
                    session, candidate_id, candidate.status
                )
                return candidate

            if attempt >= self.max_retries:
                self._append_candidate_completed(
                    session, candidate_id, candidate.status
                )
                return candidate

            self.append_session_event(
                session,
                TrialEventType.TRIAL_RETRY,
                candidate_id=candidate_id,
                metadata=TrialRetryEventMetadata(
                    attempt=attempt,
                    cand_index=cand_index,
                ),
                payload={"attempt": attempt, "cand_index": cand_index},
            )

    def _append_candidate_completed(
        self,
        session: TrialExecutionSession,
        candidate_id: str,
        status: RecordStatus,
    ) -> None:
        self.append_session_event(
            session,
            TrialEventType.CANDIDATE_COMPLETED,
            candidate_id=candidate_id,
            status=status,
            payload={"status": status.value},
        )

    def _candidate_runtime_context(
        self,
        base_runtime: RuntimeContext,
        *,
        resume_state: ResumeState | None,
        candidate_seed: int | None,
    ) -> RuntimeContext:
        return base_runtime.model_copy(
            update={"resume": resume_state, "candidate_seed": candidate_seed}
        )

    def _candidate_seed(self, trial_hash: str, cand_index: int) -> int | None:
        if self.project_seed is None:
            return None
        digest = hashlib.sha256(
            f"{self.project_seed}:{trial_hash}:{cand_index}".encode("utf-8")
        ).hexdigest()
        return int(digest[:16], 16)
