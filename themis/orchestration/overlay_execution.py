"""Internal overlay-stage execution coordinator for the trial executor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from themis.contracts.protocols import DatasetContext
from themis.errors import StorageError
from themis.orchestration.run_manifest import WorkItemStatus
from themis.progress.tracker import RunProgressTracker
from themis.orchestration.resolved_plugins import ResolvedStage
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import (
    ResolvedEvaluation,
    ResolvedOutputTransform,
    resolve_task_stages,
)
from themis.orchestration.trial_planner import PlannedTrial
from themis.orchestration.work_scheduler import WorkScheduler, WorkSchedulerStats
from themis.records.candidate import CandidateRecord
from themis.records.trial import TrialRecord
from themis.specs.experiment import ExecutionPolicySpec, RuntimeContext, TrialSpec
from themis.types.enums import ErrorCode, RecordStatus, RunStage


class _OverlayRunner(Protocol):
    """Runner operations needed for transform and evaluation overlays."""

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DatasetContext,
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        required_stages: Sequence[ResolvedStage] | None = None,
    ) -> TrialExecutionSession: ...

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


class _OverlayProjectionRepository(Protocol):
    """Projection reads needed for overlay-stage execution."""

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None: ...


@dataclass(frozen=True, slots=True)
class TransformWorkItem:
    """One output-transform request for an existing candidate."""

    session: TrialExecutionSession
    transform: ResolvedOutputTransform
    candidate: CandidateRecord


@dataclass(frozen=True, slots=True)
class EvaluationWorkItem:
    """One evaluation-stage scoring request for an existing candidate."""

    session: TrialExecutionSession
    evaluation: ResolvedEvaluation
    candidate: CandidateRecord


class OverlayExecutionCoordinator:
    """Coordinates transform/evaluation preparation, scheduling, and projection refresh."""

    def __init__(
        self,
        *,
        runner: _OverlayRunner,
        projection_repo: _OverlayProjectionRepository,
        execution_policy: ExecutionPolicySpec,
        should_skip_trial: Callable[..., bool],
        materialize_projection: Callable[..., TrialRecord | None],
        iter_trials: Callable[
            [Sequence[TrialSpec | PlannedTrial], DatasetContext | None],
            list[tuple[TrialSpec, DatasetContext]],
        ],
    ) -> None:
        self.runner = runner
        self.projection_repo = projection_repo
        self.execution_policy = execution_policy
        self.should_skip_trial = should_skip_trial
        self.materialize_projection = materialize_projection
        self.iter_trials = iter_trials

    def execute_transforms(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        progress_tracker: RunProgressTracker | None = None,
    ) -> WorkSchedulerStats:
        """Run declared output transforms against existing generation candidates."""
        prepared_sessions: list[TrialExecutionSession] = []
        base_records: dict[str, TrialRecord] = {}
        transforms_by_trial: dict[str, tuple[ResolvedOutputTransform, ...]] = {}
        for trial, trial_dataset_context in self.iter_trials(trials, dataset_context):
            resolved = resolve_task_stages(trial.task)
            pending_transforms = tuple(
                transform
                for transform in resolved.output_transforms
                if not (
                    resume
                    and self.should_skip_trial(
                        trial.spec_hash,
                        transform_hash=transform.transform_hash,
                    )
                )
            )
            if not pending_transforms:
                continue
            base_record = self.projection_repo.get_trial_record(trial.spec_hash)
            if base_record is None:
                raise StorageError(
                    code=ErrorCode.STORAGE_READ,
                    message=(
                        "Transform stage requires existing generation artifacts for "
                        f"trial {trial.trial_id}."
                    ),
                )
            prepared_sessions.append(
                _require_trial_execution_session(
                    self.runner.prepare_trial_session(
                        trial,
                        trial_dataset_context,
                        runtime_context,
                        required_stages=(RunStage.TRANSFORM,),
                    )
                )
            )
            base_records[trial.spec_hash] = base_record
            transforms_by_trial[trial.spec_hash] = pending_transforms

        scheduler = WorkScheduler(self.execution_policy.max_in_flight_work_items)
        if progress_tracker is not None and prepared_sessions:
            progress_tracker.stage_started()
        results = scheduler.run_transforms(
            (
                TransformWorkItem(
                    session=session,
                    transform=transform,
                    candidate=candidate,
                )
                for session in prepared_sessions
                for transform in transforms_by_trial[session.trial_hash]
                for candidate in base_records[session.trial_hash].candidates
            ),
            lambda work_item: self.runner.run_output_transform(
                work_item.session,
                work_item.candidate,
                work_item.transform,
            ),
            on_work_item_started=(
                lambda work_item: (
                    progress_tracker.mark_running(
                        progress_tracker.transform_work_item_id(
                            work_item.session.trial_hash,
                            work_item.candidate.sample_index,
                            work_item.transform.transform_hash,
                        )
                    )
                    if progress_tracker is not None
                    else None
                )
            ),
            on_work_item_finished=(
                lambda work_item, result, error: (
                    progress_tracker.mark_finished(
                        progress_tracker.transform_work_item_id(
                            work_item.session.trial_hash,
                            work_item.candidate.sample_index,
                            work_item.transform.transform_hash,
                        ),
                        status=(
                            WorkItemStatus.FAILED
                            if error is not None
                            or (
                                result is not None
                                and result.status == RecordStatus.ERROR
                            )
                            else WorkItemStatus.COMPLETED
                        ),
                        last_error_code=(
                            result.error.code.value
                            if result is not None and result.error is not None
                            else None
                        ),
                        last_error_message=(
                            result.error.message
                            if result is not None and result.error is not None
                            else (str(error) if error is not None else None)
                        ),
                    )
                    if progress_tracker is not None
                    else None
                )
            ),
        )

        overlays_to_materialize = {
            (
                scheduled.work_item.session.trial_hash,
                scheduled.work_item.transform.transform_hash,
            )
            for scheduled in results
        }
        for trial_hash, transform_hash in sorted(overlays_to_materialize):
            self.materialize_projection(trial_hash, transform_hash=transform_hash)
        return scheduler.last_stats

    def execute_evaluations(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        progress_tracker: RunProgressTracker | None = None,
    ) -> WorkSchedulerStats:
        """Run declared evaluations against generation or transformed candidates."""
        prepared_sessions: list[TrialExecutionSession] = []
        evaluation_inputs: dict[tuple[str, str], list[CandidateRecord]] = {}
        evaluations_by_trial: dict[str, tuple[ResolvedEvaluation, ...]] = {}
        for trial, trial_dataset_context in self.iter_trials(trials, dataset_context):
            resolved = resolve_task_stages(trial.task)
            pending_evaluations = tuple(
                evaluation
                for evaluation in resolved.evaluations
                if not (
                    resume
                    and self.should_skip_trial(
                        trial.spec_hash,
                        evaluation_hash=evaluation.evaluation_hash,
                    )
                )
            )
            if not pending_evaluations:
                continue
            prepared_sessions.append(
                _require_trial_execution_session(
                    self.runner.prepare_trial_session(
                        trial,
                        trial_dataset_context,
                        runtime_context,
                        required_stages=(RunStage.EVALUATION,),
                    )
                )
            )
            evaluations_by_trial[trial.spec_hash] = pending_evaluations
            for evaluation in pending_evaluations:
                candidate_record = self.projection_repo.get_trial_record(
                    trial.spec_hash,
                    transform_hash=(
                        evaluation.transform.transform_hash
                        if evaluation.transform is not None
                        else None
                    ),
                )
                if candidate_record is None:
                    raise StorageError(
                        code=ErrorCode.STORAGE_READ,
                        message=(
                            "Evaluation stage requires existing candidates for "
                            f"trial {trial.trial_id}, evaluation {evaluation.spec.name}."
                        ),
                    )
                evaluation_inputs[(trial.spec_hash, evaluation.evaluation_hash)] = (
                    candidate_record.candidates
                )

        scheduler = WorkScheduler(self.execution_policy.max_in_flight_work_items)
        if progress_tracker is not None and prepared_sessions:
            progress_tracker.stage_started()
        results = scheduler.run_evaluations(
            (
                EvaluationWorkItem(
                    session=session,
                    evaluation=evaluation,
                    candidate=candidate,
                )
                for session in prepared_sessions
                for evaluation in evaluations_by_trial[session.trial_hash]
                for candidate in evaluation_inputs[
                    (
                        session.trial_hash,
                        evaluation.evaluation_hash,
                    )
                ]
            ),
            lambda work_item: self.runner.run_evaluation_candidate(
                work_item.session,
                work_item.candidate,
                work_item.evaluation,
            ),
            on_work_item_started=(
                lambda work_item: (
                    progress_tracker.mark_running(
                        progress_tracker.evaluation_work_item_id(
                            work_item.session.trial_hash,
                            work_item.candidate.sample_index,
                            work_item.evaluation.evaluation_hash,
                        )
                    )
                    if progress_tracker is not None
                    else None
                )
            ),
            on_work_item_finished=(
                lambda work_item, result, error: (
                    progress_tracker.mark_finished(
                        progress_tracker.evaluation_work_item_id(
                            work_item.session.trial_hash,
                            work_item.candidate.sample_index,
                            work_item.evaluation.evaluation_hash,
                        ),
                        status=(
                            WorkItemStatus.FAILED
                            if error is not None
                            or (
                                result is not None
                                and result.status == RecordStatus.ERROR
                            )
                            else WorkItemStatus.COMPLETED
                        ),
                        last_error_code=(
                            result.error.code.value
                            if result is not None and result.error is not None
                            else None
                        ),
                        last_error_message=(
                            result.error.message
                            if result is not None and result.error is not None
                            else (str(error) if error is not None else None)
                        ),
                    )
                    if progress_tracker is not None
                    else None
                )
            ),
        )

        overlays_to_materialize = {
            (
                scheduled.work_item.session.trial_hash,
                scheduled.work_item.evaluation.evaluation_hash,
            )
            for scheduled in results
        }
        for trial_hash, evaluation_hash in sorted(overlays_to_materialize):
            self.materialize_projection(trial_hash, evaluation_hash=evaluation_hash)
        return scheduler.last_stats


def _require_trial_execution_session(
    session: TrialExecutionSession,
) -> TrialExecutionSession:
    if not isinstance(session, TrialExecutionSession):
        raise TypeError(
            "Execution runner must return a TrialExecutionSession from "
            "prepare_trial_session()."
        )
    return session
