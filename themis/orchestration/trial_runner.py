"""Trial execution pipeline that records typed lifecycle events."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Mapping
from platform import platform as platform_name
from sys import version as python_version

from pydantic import TypeAdapter

from themis._version import __version__
from themis._replay import CandidateReplayState, ResumeState
from themis.contracts.protocols import DatasetContext, TrialEventRepository
from themis.errors.mapping import map_exception_to_error_record
from themis.errors.exceptions import SpecValidationError
from themis.orchestration.candidate_pipeline import (
    candidate_hash_for_index,
    execute_candidate_pipeline,
)
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, ConversationEvent
from themis.records.error import ErrorRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.provenance import ProvenanceRecord
from themis.records.trial import TrialRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import RuntimeContext, TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode, ErrorWhere, RecordStatus
from themis.types.events import (
    ArtifactRef,
    ArtifactRole,
    TimelineStage,
    TrialEvent,
    TrialEventType,
)
from themis.types.json_types import JSONDict, JSONValueType
from themis.types.json_validation import dump_storage_json_bytes, validate_json_value

_CONVERSATION_EVENT_ADAPTER: TypeAdapter[ConversationEvent] = TypeAdapter(
    ConversationEvent
)


def _json_value(data: object, *, label: str) -> JSONValueType:
    return validate_json_value(data, label=label)


def _artifact_ref(
    data: object,
    *,
    role: ArtifactRole,
    label: str,
    artifact_store: ArtifactStore | None = None,
) -> tuple[ArtifactRef, str]:
    payload = dump_storage_json_bytes(data, label=label)

    artifact_hash = (
        artifact_store.put_blob(payload, "application/json")
        if artifact_store is not None
        else f"sha256:{hashlib.sha256(payload).hexdigest()}"
    )
    return (
        ArtifactRef(
            artifact_hash=artifact_hash,
            media_type="application/json",
            label=label,
            role=role,
        ),
        artifact_hash,
    )


def _dataset_payload(dataset_context: DatasetContext) -> dict[str, object]:
    if hasattr(dataset_context, "payload"):
        payload = getattr(dataset_context, "payload")
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(dataset_context, Mapping):
        return {str(key): value for key, value in dataset_context.items()}
    return {}


class TrialRunner:
    """Executes one trial and emits the full typed lifecycle event stream."""

    def __init__(
        self,
        registry: PluginRegistry,
        event_repo: TrialEventRepository,
        artifact_store: ArtifactStore | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.5,
        parallel_candidates: int = 5,
        project_seed: int | None = None,
        store_item_payloads: bool = True,
        telemetry_bus: TelemetryBus | None = None,
    ):
        if parallel_candidates < 1:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message="parallel_candidates must be >= 1.",
            )
        self.registry = registry
        self.event_repo = event_repo
        self.artifact_store = artifact_store
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.parallel_candidates = parallel_candidates
        self.project_seed = project_seed
        self.store_item_payloads = store_item_payloads
        self.telemetry_bus = telemetry_bus

    def run_trial(
        self,
        trial: TrialSpec,
        dataset_context: DatasetContext,
        runtime_context: RuntimeContext | Mapping[str, object] | None,
    ) -> TrialRecord:
        """Execute one trial and persist its event stream before projection.

        Args:
            trial: Expanded trial specification to execute.
            dataset_context: Materialized dataset row for the trial item.
            runtime_context: Optional runtime overrides or resume metadata.

        Returns:
            The fully materialized trial record reconstructed from emitted events.
        """
        trial_hash = trial.spec_hash
        provenance = self._build_provenance()
        base_runtime = self._coerce_runtime_context(runtime_context)
        self.event_repo.save_spec(trial)
        event_seq = self.event_repo.last_event_index(trial_hash) or 0
        event_lock = threading.Lock()
        existing_trial_events = self._get_trial_events(trial_hash)

        def append_event(
            event_type: TrialEventType,
            *,
            candidate_id: str | None = None,
            stage: TimelineStage | None = None,
            status: RecordStatus | None = None,
            metadata: JSONDict | None = None,
            payload: JSONValueType | None = None,
            artifact_refs: list[ArtifactRef] | None = None,
            error: ErrorRecord | None = None,
        ) -> None:
            nonlocal event_seq
            with event_lock:
                event_seq += 1
                self.event_repo.append_event(
                    TrialEvent(
                        trial_hash=trial_hash,
                        event_seq=event_seq,
                        event_id=f"{trial_hash}:{event_seq}",
                        candidate_id=candidate_id,
                        event_type=event_type,
                        stage=stage,
                        status=status,
                        metadata=metadata or {},
                        payload=payload,
                        artifact_refs=artifact_refs or [],
                        error=error,
                    )
                )
                self._emit_telemetry_event(
                    TrialEvent(
                        trial_hash=trial_hash,
                        event_seq=event_seq,
                        event_id=f"{trial_hash}:{event_seq}",
                        candidate_id=candidate_id,
                        event_type=event_type,
                        stage=stage,
                        status=status,
                        metadata=metadata or {},
                        payload=payload,
                        artifact_refs=artifact_refs or [],
                        error=error,
                    )
                )

        prompt_payload = _json_value(
            {
                "messages": [
                    message.model_dump(mode="json") for message in trial.prompt.messages
                ]
            },
            label="rendered prompt",
        )
        item_payload = _json_value(
            _dataset_payload(dataset_context), label="dataset payload"
        )
        dataset_metadata = getattr(dataset_context, "metadata", {})
        prompt_artifact = _artifact_ref(
            prompt_payload,
            role=ArtifactRole.RENDERED_PROMPT,
            label="rendered_prompt",
            artifact_store=self.artifact_store,
        )

        if not existing_trial_events:
            append_event(
                TrialEventType.TRIAL_STARTED, payload={"trial_id": trial.trial_id}
            )
            item_artifact = (
                _artifact_ref(
                    item_payload,
                    role=ArtifactRole.ITEM_PAYLOAD,
                    label="item_payload",
                    artifact_store=self.artifact_store,
                )
                if self.store_item_payloads
                else None
            )
            item_metadata: JSONDict = {
                "item_id": trial.item_id,
                "dataset_source": trial.task.dataset.source,
                "dataset_revision": trial.task.dataset.revision,
            }
            tags: dict[str, str] = {}
            if isinstance(dataset_metadata, Mapping):
                tags.update(
                    {str(key): str(value) for key, value in dataset_metadata.items()}
                )
            tags.update(base_runtime.run_labels)
            if tags:
                item_metadata["tags"] = _json_value(tags, label="item tags")
            if item_artifact is not None:
                item_metadata["item_payload_hash"] = item_artifact[1]
            prompt_metadata: JSONDict = {
                "prompt_template_id": trial.prompt.id,
                "rendered_prompt_hash": prompt_artifact[1],
                "input_field_map": _json_value(
                    sorted(dataset_context.keys()),
                    label="prompt input fields",
                ),
            }
            append_event(
                TrialEventType.ITEM_LOADED,
                stage="item_load",
                status=RecordStatus.OK,
                metadata=item_metadata,
                payload=item_payload if self.store_item_payloads else None,
                artifact_refs=[item_artifact[0]] if item_artifact is not None else [],
            )
            append_event(
                TrialEventType.PROMPT_RENDERED,
                stage="prompt_render",
                status=RecordStatus.OK,
                metadata=prompt_metadata,
                payload=prompt_payload,
                artifact_refs=[prompt_artifact[0]],
            )

        async def run_single_candidate(cand_index: int) -> CandidateRecord:
            from themis.evaluation.judge_service import DefaultJudgeService

            candidate_id = candidate_hash_for_index(trial, cand_index)
            candidate_events = self._get_trial_events(
                trial_hash, candidate_id=candidate_id
            )
            terminal_candidate = self._candidate_from_terminal_events(
                candidate_id,
                cand_index,
                candidate_events,
            )
            if terminal_candidate is not None:
                return terminal_candidate
            resume_state = self._resume_state(candidate_id, candidate_events)

            if not candidate_events:
                prompt_metadata: JSONDict = {
                    "prompt_template_id": trial.prompt.id,
                    "rendered_prompt_hash": prompt_artifact[1],
                    "input_field_map": _json_value(
                        sorted(dataset_context.keys()),
                        label="prompt input fields",
                    ),
                }
                append_event(
                    TrialEventType.CANDIDATE_STARTED,
                    candidate_id=candidate_id,
                    payload={"sample_index": cand_index},
                )
                append_event(
                    TrialEventType.PROMPT_RENDERED,
                    candidate_id=candidate_id,
                    stage="prompt_render",
                    status=RecordStatus.OK,
                    metadata=prompt_metadata,
                    payload=prompt_payload,
                    artifact_refs=[prompt_artifact[0]],
                )

            attempt = 0
            while True:
                attempt += 1
                judge_service = DefaultJudgeService(self.registry)
                candidate_runtime = self._candidate_runtime_context(
                    base_runtime,
                    resume_state=resume_state,
                    candidate_seed=self._candidate_seed(trial, cand_index),
                )
                try:
                    candidate = await asyncio.to_thread(
                        execute_candidate_pipeline,
                        trial,
                        self.registry,
                        dataset_context,
                        candidate_runtime,
                        cand_index,
                        judge_service,
                        self.telemetry_bus,
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
                        provenance=provenance,
                    )
                    if self.telemetry_bus is not None:
                        self.telemetry_bus.emit(
                            "error",
                            trial_hash=trial_hash,
                            candidate_id=candidate_id,
                            provider=trial.model.provider,
                            model_id=trial.model.model_id,
                            code=error.code.value,
                            where=error.where.value,
                            message=error.message,
                        )
                judge_trails = self._collect_judge_trails(judge_service, candidate_id)
                judge_artifact_refs = self._persist_judge_artifacts(judge_trails)
                candidate = self._apply_judge_artifacts(
                    candidate, judge_artifact_refs
                ).model_copy(update={"provenance": provenance})
                self._emit_candidate_stage_events(
                    trial,
                    candidate,
                    append_event,
                    resume_state=resume_state,
                    judge_artifact_refs=judge_artifact_refs,
                )

                if not (
                    candidate.status == RecordStatus.ERROR
                    and candidate.error
                    and candidate.error.retryable
                ):
                    append_event(
                        TrialEventType.CANDIDATE_COMPLETED,
                        candidate_id=candidate_id,
                        status=candidate.status,
                        payload={"status": candidate.status.value},
                    )
                    return candidate

                if attempt >= self.max_retries:
                    append_event(
                        TrialEventType.CANDIDATE_COMPLETED,
                        candidate_id=candidate_id,
                        status=candidate.status,
                        payload={"status": candidate.status.value},
                    )
                    return candidate

                append_event(
                    TrialEventType.TRIAL_RETRY,
                    candidate_id=candidate_id,
                    payload={"attempt": attempt, "cand_index": cand_index},
                )

        async def run_all_candidates() -> list[CandidateRecord]:
            semaphore = asyncio.Semaphore(
                min(self.parallel_candidates, trial.candidate_count)
            )

            async def run_with_semaphore(index: int) -> CandidateRecord:
                async with semaphore:
                    return await run_single_candidate(index)

            return await asyncio.gather(
                *(run_with_semaphore(index) for index in range(trial.candidate_count))
            )

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            result_container: list[list[CandidateRecord]] = []

            def thread_target() -> None:
                result_container.append(asyncio.run(run_all_candidates()))

            worker = threading.Thread(target=thread_target)
            worker.start()
            worker.join()
            candidates = result_container[0]
        else:
            candidates = asyncio.run(run_all_candidates())

        overall_status = RecordStatus.OK
        trial_error = None
        for candidate in candidates:
            if candidate.status == RecordStatus.ERROR:
                overall_status = RecordStatus.ERROR
                trial_error = candidate.error
                break

        trial_record = TrialRecord(
            spec_hash=trial_hash,
            trial_spec=trial,
            status=overall_status,
            error=trial_error,
            candidates=candidates,
            provenance=provenance,
        )

        if overall_status == RecordStatus.ERROR:
            append_event(
                TrialEventType.TRIAL_FAILED,
                status=overall_status,
                payload={"status": overall_status.value},
                error=trial_error,
            )
        else:
            append_event(
                TrialEventType.TRIAL_COMPLETED,
                status=overall_status,
                payload={"status": overall_status.value},
            )

        return trial_record

    def _emit_candidate_stage_events(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        append_event,
        *,
        resume_state: ResumeState | None = None,
        judge_artifact_refs: list[ArtifactRef] | None = None,
    ) -> None:
        candidate_id = candidate.spec_hash
        resume_index = (
            resume_state.last_event_index if resume_state is not None else None
        )

        if candidate.inference is not None:
            token_usage = candidate.inference.token_usage
            inference_payload = candidate.inference.model_dump(mode="json")
            inference_artifact = _artifact_ref(
                inference_payload,
                role=ArtifactRole.INFERENCE_OUTPUT,
                label="inference_output",
                artifact_store=self.artifact_store,
            )
            append_event(
                TrialEventType.INFERENCE_COMPLETED,
                candidate_id=candidate_id,
                stage="inference",
                status=RecordStatus.OK,
                metadata={
                    "provider": trial.model.provider,
                    "model_id": trial.model.model_id,
                    "inference_params_hash": trial.params.spec_hash,
                    "provider_request_id": candidate.inference.provider_request_id,
                    "token_usage": (
                        token_usage.model_dump(mode="json")
                        if token_usage is not None
                        else {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    ),
                },
                payload=inference_payload,
                artifact_refs=[inference_artifact[0]],
            )
            if candidate.conversation is not None:
                for conversation_event in candidate.conversation.events:
                    if (
                        resume_index is not None
                        and conversation_event.event_index <= resume_index
                    ):
                        continue
                    append_event(
                        TrialEventType.CONVERSATION_EVENT,
                        candidate_id=candidate_id,
                        payload=conversation_event.model_dump(mode="json"),
                    )

        for attempt_index, extraction in enumerate(candidate.extractions):
            extraction_payload = extraction.model_dump(mode="json")
            append_event(
                TrialEventType.EXTRACTION_COMPLETED,
                candidate_id=candidate_id,
                stage="extraction",
                status=RecordStatus.OK if extraction.success else RecordStatus.ERROR,
                metadata={
                    "extractor_id": extraction.extractor_id,
                    "attempt_index": attempt_index,
                    "success": extraction.success,
                    "failure_reason": extraction.failure_reason,
                },
                payload=extraction_payload,
            )

        if candidate.evaluation is not None and candidate.evaluation.metric_scores:
            first_score = candidate.evaluation.metric_scores[0]
            details_artifact = (
                _artifact_ref(
                    first_score.details,
                    role=ArtifactRole.METRIC_DETAILS,
                    label="metric_details",
                    artifact_store=self.artifact_store,
                )
                if first_score.details
                else None
            )
            append_event(
                TrialEventType.EVALUATION_COMPLETED,
                candidate_id=candidate_id,
                stage="evaluation",
                status=RecordStatus.OK,
                metadata={
                    "metric_id": first_score.metric_id,
                    "score": first_score.value,
                    "judge_call_count": len(candidate.judge_audits),
                    "details_hash": details_artifact[1]
                    if details_artifact is not None
                    else None,
                    "judge_audit_hashes": [
                        artifact.artifact_hash for artifact in judge_artifact_refs or []
                    ],
                },
                payload=candidate.evaluation.model_dump(mode="json"),
                artifact_refs=[
                    *([details_artifact[0]] if details_artifact is not None else []),
                    *(judge_artifact_refs or []),
                ],
            )

        if candidate.status == RecordStatus.ERROR and candidate.error is not None:
            stage = self._stage_for_error(candidate.error.where)
            append_event(
                TrialEventType.CANDIDATE_FAILED,
                candidate_id=candidate_id,
                stage=stage,
                status=RecordStatus.ERROR,
                error=candidate.error,
            )

    def _stage_for_error(self, where: ErrorWhere) -> str:
        if where == ErrorWhere.EXTRACTOR:
            return "extraction"
        if where == ErrorWhere.METRIC:
            return "evaluation"
        return "inference"

    def _get_trial_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        return list(self.event_repo.get_events(trial_hash, candidate_id=candidate_id))

    def _resume_state(
        self,
        candidate_id: str,
        candidate_events: list[TrialEvent],
    ) -> ResumeState | None:
        if not candidate_events:
            return None

        conversation_events = [
            _CONVERSATION_EVENT_ADAPTER.validate_python(event.payload)
            for event in candidate_events
            if event.event_type == TrialEventType.CONVERSATION_EVENT
            and event.payload is not None
        ]
        if not conversation_events:
            return None

        return ResumeState(
            candidate_id=candidate_id,
            conversation=Conversation(events=conversation_events),
            last_event_index=conversation_events[-1].event_index,
        )

    def _candidate_from_terminal_events(
        self,
        candidate_id: str,
        sample_index: int,
        candidate_events: list[TrialEvent],
    ) -> CandidateRecord | None:
        terminal_events = [
            event
            for event in candidate_events
            if event.event_type
            in {TrialEventType.CANDIDATE_COMPLETED, TrialEventType.CANDIDATE_FAILED}
        ]
        if not terminal_events:
            return None

        state = CandidateReplayState(sample_index=sample_index)

        for event in candidate_events:
            state.apply_event(event)

        return state.to_candidate_record(candidate_id)

    def _candidate_runtime_context(
        self,
        runtime_context: RuntimeContext | Mapping[str, object] | None,
        *,
        resume_state: ResumeState | None,
        candidate_seed: int | None,
    ) -> RuntimeContext:
        runtime = self._coerce_runtime_context(runtime_context)
        return runtime.model_copy(
            update={"resume": resume_state, "candidate_seed": candidate_seed}
        )

    def _coerce_runtime_context(
        self,
        runtime_context: RuntimeContext | Mapping[str, object] | None,
    ) -> RuntimeContext:
        if runtime_context is None:
            return RuntimeContext()
        if isinstance(runtime_context, RuntimeContext):
            return runtime_context
        if isinstance(runtime_context, Mapping):
            return RuntimeContext.model_validate(dict(runtime_context))
        raise TypeError("runtime_context must be a RuntimeContext, mapping, or None.")

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

    def _candidate_seed(self, trial: TrialSpec, cand_index: int) -> int | None:
        if self.project_seed is None:
            return None
        digest = hashlib.sha256(
            f"{self.project_seed}:{trial.spec_hash}:{cand_index}".encode("utf-8")
        ).hexdigest()
        return int(digest[:16], 16)

    def _build_provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            themis_version=__version__,
            git_commit=None,
            python_version=python_version.split()[0],
            platform=platform_name(),
            library_versions={},
            model_endpoint_meta={},
        )

    def _emit_telemetry_event(self, event: TrialEvent) -> None:
        if self.telemetry_bus is None:
            return
        payload: dict[str, JSONValueType] = {
            "trial_hash": event.trial_hash,
        }
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
            self.telemetry_bus.emit("trial_start", **payload)
            return
        if event.event_type in {
            TrialEventType.TRIAL_COMPLETED,
            TrialEventType.TRIAL_FAILED,
        }:
            self.telemetry_bus.emit("trial_end", **payload)
            if event.error is not None:
                self.telemetry_bus.emit("error", **payload)
            return
        if event.event_type == TrialEventType.CONVERSATION_EVENT:
            self.telemetry_bus.emit("conversation_event", **payload)
            kind = (
                event.payload.get("kind") if isinstance(event.payload, dict) else None
            )
            if kind == "tool_call":
                self.telemetry_bus.emit("tool_call", **payload)
            elif kind == "tool_result":
                self.telemetry_bus.emit("tool_result", **payload)
            return
        if (
            event.event_type == TrialEventType.CANDIDATE_FAILED
            and event.error is not None
        ):
            self.telemetry_bus.emit("error", **payload)
