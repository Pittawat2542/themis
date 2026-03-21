"""Trial execution pipeline that records typed lifecycle events."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from platform import platform as platform_name
from sys import version as python_version

from themis._version import __version__
from themis.contracts.protocols import DatasetContext, TrialEventRepository
from themis.errors import SpecValidationError
from themis.orchestration._async import run_coroutine_sync
from themis.orchestration import candidate_pipeline
from themis.orchestration.resolved_plugins import resolve_trial_plugins
from themis.orchestration.generation_stage import GenerationStageExecutor
from themis.orchestration.overlay_stage import OverlayStageExecutor
from themis.orchestration.runner_events import TrialEventEmitter
from themis.orchestration.session_preparer import TrialSessionPreparer
from themis.orchestration.session_preparer import prepare_benchmark_prompt
from themis.orchestration.runner_state import (
    TrialExecutionSession,
)
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
)
from themis.orchestration.trial_finalizer import TrialFinalizer
from themis.records.candidate import CandidateRecord
from themis.records.provenance import ProvenanceRecord
from themis.records.trial import TrialRecord
from themis.registry.compatibility import resolve_runtime_tool_handlers
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import RuntimeContext, ToolHandler, TrialSpec
from themis.storage.artifact_store import ArtifactStore
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import (
    ArtifactRef,
    TrialEvent,
    TrialEventMetadata,
    TrialEventType,
)
from themis.types.json_types import JSONValueType
from themis.orchestration.resolved_plugins import ResolvedStage

candidate_hash_for_index = candidate_pipeline.candidate_hash_for_index


class TrialRunner:
    """Executes one trial and emits the full typed lifecycle event stream."""

    def __init__(
        self,
        registry: PluginRegistry,
        event_repo: TrialEventRepository,
        artifact_store: ArtifactStore | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.5,
        retryable_error_codes: list[str] | None = None,
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
        invalid_error_codes: list[str] = []
        validated_error_codes: list[ErrorCode] = []
        for value in retryable_error_codes or []:
            try:
                validated_error_codes.append(ErrorCode(value))
            except ValueError:
                invalid_error_codes.append(value)
        if invalid_error_codes:
            invalid_values = ", ".join(
                sorted(repr(value) for value in invalid_error_codes)
            )
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(f"Unknown retryable_error_codes: {invalid_values}."),
            )
        self.retryable_error_codes = tuple(validated_error_codes)
        self.parallel_candidates = parallel_candidates
        self.project_seed = project_seed
        self.store_item_payloads = store_item_payloads
        self.telemetry_bus = telemetry_bus
        self.event_emitter = TrialEventEmitter(
            event_repo,
            artifact_store=artifact_store,
            telemetry_bus=telemetry_bus,
        )
        self.session_preparer = TrialSessionPreparer(
            event_repo=event_repo,
            artifact_store=artifact_store,
            store_item_payloads=store_item_payloads,
            append_session_event=self._append_session_event,
        )
        self.trial_finalizer = TrialFinalizer(
            append_session_event=self._append_session_event,
        )
        self.generation_stage = GenerationStageExecutor(
            registry=registry,
            event_emitter=self.event_emitter,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            retryable_error_codes=self.retryable_error_codes,
            project_seed=project_seed,
            telemetry_bus=telemetry_bus,
            append_session_event=self._append_session_event,
            get_trial_events=self._get_trial_events,
        )
        self.overlay_stage = OverlayStageExecutor(
            registry=registry,
            event_emitter=self.event_emitter,
            artifact_store=artifact_store,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            retryable_error_codes=self.retryable_error_codes,
            telemetry_bus=telemetry_bus,
            append_session_event=self._append_session_event,
        )

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DatasetContext,
        runtime_context: RuntimeContext | Mapping[str, object] | None,
        *,
        required_stages: Collection[ResolvedStage] | None = None,
    ) -> TrialExecutionSession:
        """Build the shared execution context for one trial."""
        provenance = self._build_provenance()
        base_runtime = self._coerce_runtime_context(runtime_context)
        resolved_plugins = resolve_trial_plugins(
            trial,
            self.registry,
            resolved_stages=None,
            required_stages=tuple(required_stages)
            if required_stages is not None
            else None,
        )
        prepared_trial = resolved_plugins.hooks.apply_pre_inference(
            prepare_benchmark_prompt(trial, dataset_context, base_runtime)
        )
        prepared_runtime = base_runtime.model_copy(
            update={
                "tool_handlers": self._resolve_tool_handlers(
                    prepared_trial,
                    base_runtime=base_runtime,
                )
            }
        )
        session = self.session_preparer.prepare_trial_session(
            trial,
            prepared_trial,
            dataset_context,
            prepared_runtime,
            provenance,
        )
        session.resolved_plugins = resolved_plugins
        return session

    def run_generation_candidate(
        self,
        session: TrialExecutionSession,
        cand_index: int,
    ) -> CandidateRecord:
        """Execute the generation stage for one candidate within a prepared trial."""
        return self.generation_stage.run_candidate(session, cand_index)

    def finalize_generation_trial(
        self,
        session: TrialExecutionSession,
        candidates: list[CandidateRecord],
    ) -> TrialRecord:
        """Emit terminal trial events and build the generation-scoped trial record."""
        return self.trial_finalizer.finalize_generation_trial(session, candidates)

    def run_output_transform(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        transform: ResolvedOutputTransform,
    ) -> CandidateRecord:
        """Apply one output transform and append overlay-specific extraction events."""
        return self.overlay_stage.run_output_transform(session, candidate, transform)

    def run_evaluation_candidate(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        evaluation: ResolvedEvaluation,
    ) -> CandidateRecord:
        """Apply one evaluation and append overlay-specific scoring events."""
        return self.overlay_stage.run_evaluation_candidate(
            session, candidate, evaluation
        )

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
        session = self.prepare_trial_session(trial, dataset_context, runtime_context)

        async def run_single_candidate(cand_index: int) -> CandidateRecord:
            return await asyncio.to_thread(
                self._run_trial_candidate_pipeline,
                session,
                cand_index,
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

        candidates = run_coroutine_sync(run_all_candidates)
        return self.finalize_generation_trial(session, candidates)

    def _run_trial_candidate_pipeline(
        self,
        session: TrialExecutionSession,
        cand_index: int,
    ) -> CandidateRecord:
        generated_candidate = self.run_generation_candidate(session, cand_index)
        if generated_candidate.status == RecordStatus.ERROR:
            return generated_candidate

        transformed_candidates: dict[str, CandidateRecord] = {}
        first_transform_candidate: CandidateRecord | None = None
        for transform in session.resolved_stages.output_transforms:
            transformed_candidate = self.run_output_transform(
                session,
                generated_candidate,
                transform,
            )
            transformed_candidates[transform.transform_hash] = transformed_candidate
            if first_transform_candidate is None:
                first_transform_candidate = transformed_candidate
            if transformed_candidate.status == RecordStatus.ERROR:
                return transformed_candidate

        if not session.resolved_stages.evaluations:
            return first_transform_candidate or generated_candidate

        final_candidate = generated_candidate
        for evaluation in session.resolved_stages.evaluations:
            candidate_view = generated_candidate
            if evaluation.transform is not None:
                candidate_view = transformed_candidates[
                    evaluation.transform.transform_hash
                ]
            evaluated_candidate = self.run_evaluation_candidate(
                session,
                candidate_view,
                evaluation,
            )
            if evaluated_candidate.status == RecordStatus.ERROR:
                return evaluated_candidate
            final_candidate = evaluated_candidate
        return final_candidate

    def _append_session_event(
        self,
        session: TrialExecutionSession,
        event_type: TrialEventType,
        *,
        candidate_id: str | None = None,
        stage=None,
        status: RecordStatus | None = None,
        metadata: TrialEventMetadata | None = None,
        payload: JSONValueType | None = None,
        artifact_refs: list[ArtifactRef] | None = None,
        error=None,
    ) -> None:
        self.event_emitter.append_session_event(
            session,
            event_type,
            candidate_id=candidate_id,
            stage=stage,
            status=status,
            metadata=metadata,
            payload=payload,
            artifact_refs=artifact_refs,
            error=error,
        )

    def _get_trial_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        return list(self.event_repo.get_events(trial_hash, candidate_id=candidate_id))

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

    def _resolve_tool_handlers(
        self,
        trial: TrialSpec,
        *,
        base_runtime: RuntimeContext,
    ) -> dict[str, ToolHandler]:
        return resolve_runtime_tool_handlers(
            trial,
            self.registry,
            runtime_handlers=base_runtime.tool_handlers,
        )

    def _build_provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            themis_version=__version__,
            git_commit=None,
            python_version=python_version.split()[0],
            platform=platform_name(),
            library_versions={},
            model_endpoint_meta={},
        )
