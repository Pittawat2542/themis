"""Trial execution coordinator with bounded global scheduling and overlays."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from themis.contracts.protocols import (
    DatasetContext,
    ProjectionHandler,
    TrialEventRepository,
)
from themis.progress.tracker import RunProgressTracker
from themis.orchestration._executor_support import ExecutionSupport
from themis.orchestration.generation_execution import GenerationExecutionCoordinator
from themis.orchestration.overlay_execution import OverlayExecutionCoordinator
from themis.orchestration.resolved_plugins import ResolvedStage
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
)
from themis.orchestration.trial_planner import PlannedTrial
from themis.orchestration.work_scheduler import WorkSchedulerStats
from themis.records.candidate import CandidateRecord
from themis.records.trial import TrialRecord
from themis.specs.experiment import (
    ExecutionPolicySpec,
    RuntimeContext,
    TrialSpec,
)
from themis.telemetry.bus import TelemetryBus

logger = logging.getLogger(__name__)


class _ExecutionRunner(Protocol):
    """Internal runner contract used by the executor's staged scheduler."""

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DatasetContext,
        runtime_context: RuntimeContext | Mapping[str, object] | None,
        *,
        required_stages: Sequence[ResolvedStage] | None = None,
    ) -> TrialExecutionSession: ...

    def run_generation_candidate(
        self,
        session: TrialExecutionSession,
        cand_index: int,
    ) -> CandidateRecord: ...

    def finalize_generation_trial(
        self,
        session: TrialExecutionSession,
        candidates: list[CandidateRecord],
    ) -> TrialRecord: ...

    def run_output_transform(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        transform: ResolvedOutputTransform,
    ) -> CandidateRecord: ...

    def run_evaluation_candidate(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        evaluation: ResolvedEvaluation,
    ) -> CandidateRecord: ...


class _ExecutionProjectionRepository(Protocol):
    """Read-side projection contract needed during staged execution."""

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None: ...


class TrialExecutor:
    """Coordinates stage execution, resume checks, projections, and circuit breaking."""

    def __init__(
        self,
        runner: _ExecutionRunner,
        projection_repo: _ExecutionProjectionRepository,
        event_repo: TrialEventRepository | None = None,
        projection_handler: ProjectionHandler | None = None,
        execution_policy: ExecutionPolicySpec | None = None,
        telemetry_bus: TelemetryBus | None = None,
    ) -> None:
        self.execution_policy = execution_policy or ExecutionPolicySpec()
        self.last_scheduler_stats: WorkSchedulerStats | None = None
        self._support = ExecutionSupport(
            projection_repo=projection_repo,
            event_repo=event_repo,
            projection_handler=projection_handler,
            execution_policy=self.execution_policy,
            telemetry_bus=telemetry_bus,
        )
        self._generation_execution = GenerationExecutionCoordinator(
            runner=runner,
            execution_policy=self.execution_policy,
            should_skip_trial=self._support.should_skip_trial,
            iter_trials=lambda trials, dataset_context: self._iter_trials(
                trials,
                dataset_context=dataset_context,
            ),
            materialize_projection=self._support.materialize_projection,
            update_circuit_breaker=self._support.update_circuit_breaker,
            record_terminal_failure=self._support.record_terminal_failure,
        )
        self._overlay_execution = OverlayExecutionCoordinator(
            runner=runner,
            projection_repo=projection_repo,
            execution_policy=self.execution_policy,
            should_skip_trial=self._support.should_skip_trial,
            materialize_projection=self._support.materialize_projection,
            iter_trials=lambda trials, dataset_context: self._iter_trials(
                trials,
                dataset_context=dataset_context,
            ),
        )

    def execute_generation_trials(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        progress_tracker: RunProgressTracker | None = None,
    ) -> None:
        """Run generation work items with one bounded global scheduler."""
        for trial, _trial_dataset_context in self._iter_trials(
            trials,
            dataset_context=dataset_context,
        ):
            if resume and self._support.should_skip_trial(trial.spec_hash):
                logger.info("Skipping cached generation: %s", trial.trial_id)
            else:
                logger.info("Executing generation for trial: %s", trial.trial_id)
        kwargs = {
            "dataset_context": dataset_context,
            "resume": resume,
        }
        if progress_tracker is not None:
            kwargs["progress_tracker"] = progress_tracker
        self.last_scheduler_stats = (
            self._generation_execution.execute_generation_trials(
                trials,
                runtime_context,
                **kwargs,
            )
        )

    def execute_transforms(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        progress_tracker: RunProgressTracker | None = None,
    ) -> None:
        """Run declared output transforms against existing generation candidates."""
        kwargs = {
            "dataset_context": dataset_context,
            "resume": resume,
        }
        if progress_tracker is not None:
            kwargs["progress_tracker"] = progress_tracker
        self.last_scheduler_stats = self._overlay_execution.execute_transforms(
            trials,
            runtime_context,
            **kwargs,
        )

    def execute_evaluations(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        progress_tracker: RunProgressTracker | None = None,
    ) -> None:
        """Run declared evaluations against generation or transformed candidates."""
        kwargs = {
            "dataset_context": dataset_context,
            "resume": resume,
        }
        if progress_tracker is not None:
            kwargs["progress_tracker"] = progress_tracker
        self.last_scheduler_stats = self._overlay_execution.execute_evaluations(
            trials,
            runtime_context,
            **kwargs,
        )

    def _iter_trials(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        *,
        dataset_context: DatasetContext | None,
    ) -> list[tuple[TrialSpec, DatasetContext]]:
        return self._support.iter_trials(
            trials,
            dataset_context=dataset_context,
        )
