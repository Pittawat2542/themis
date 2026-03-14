"""Internal generation-stage execution coordinator for the trial executor."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from themis.contracts.protocols import DatasetContext
from themis.orchestration.resolved_plugins import ResolvedStage
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.trial_planner import PlannedTrial
from themis.orchestration.work_scheduler import WorkScheduler, WorkSchedulerStats
from themis.records.candidate import CandidateRecord
from themis.records.trial import TrialRecord
from themis.specs.experiment import ExecutionPolicySpec, RuntimeContext, TrialSpec


class _GenerationRunner(Protocol):
    """Runner operations needed for generation-stage execution."""

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DatasetContext,
        runtime_context: Mapping[str, object] | RuntimeContext | None,
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


@dataclass(frozen=True, slots=True)
class GenerationWorkItem:
    """One generation-stage candidate execution request."""

    session: TrialExecutionSession
    candidate_index: int


class GenerationExecutionCoordinator:
    """Coordinates generation-stage preparation, scheduling, and finalization."""

    def __init__(
        self,
        *,
        runner: _GenerationRunner,
        execution_policy: ExecutionPolicySpec,
        should_skip_trial: Callable[..., bool],
        iter_trials: Callable[
            [Sequence[TrialSpec | PlannedTrial], DatasetContext | None],
            list[tuple[TrialSpec, DatasetContext]],
        ],
        materialize_projection: Callable[..., TrialRecord | None],
        update_circuit_breaker: Callable[[TrialRecord], None],
        record_terminal_failure: Callable[[TrialSpec, Exception], None],
    ) -> None:
        self.runner = runner
        self.execution_policy = execution_policy
        self.should_skip_trial = should_skip_trial
        self.iter_trials = iter_trials
        self.materialize_projection = materialize_projection
        self.update_circuit_breaker = update_circuit_breaker
        self.record_terminal_failure = record_terminal_failure

    def execute_generation_trials(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
    ) -> WorkSchedulerStats:
        """Run generation work items with one bounded global scheduler."""
        prepared_sessions: list[TrialExecutionSession] = []
        for trial, trial_dataset_context in self.iter_trials(trials, dataset_context):
            if resume and self.should_skip_trial(trial.spec_hash):
                continue
            prepared_sessions.append(
                _require_trial_execution_session(
                    self.runner.prepare_trial_session(
                        trial,
                        trial_dataset_context,
                        runtime_context,
                        required_stages=("generation",),
                    )
                )
            )

        scheduler = WorkScheduler(self.execution_policy.max_in_flight_work_items)
        results = scheduler.run_generation(
            self._generation_work_items(prepared_sessions),
            lambda work_item: self.runner.run_generation_candidate(
                work_item.session,
                work_item.candidate_index,
            ),
        )

        candidates_by_trial: dict[str, list[CandidateRecord]] = defaultdict(list)
        for scheduled in results:
            candidates_by_trial[scheduled.work_item.session.trial_hash].append(
                scheduled.result
            )

        for session in prepared_sessions:
            try:
                trial_record = self.runner.finalize_generation_trial(
                    session,
                    candidates_by_trial.get(session.trial_hash, []),
                )
                projected_record = self.materialize_projection(session.trial_hash)
                self.update_circuit_breaker(projected_record or trial_record)
            except Exception as exc:
                self.record_terminal_failure(session.trial, exc)
                raise
        return scheduler.last_stats

    def _generation_work_items(
        self,
        sessions: Sequence[TrialExecutionSession],
    ):
        max_candidate_count = max(
            (session.trial.candidate_count for session in sessions),
            default=0,
        )
        for candidate_index in range(max_candidate_count):
            for session in sessions:
                if candidate_index < session.trial.candidate_count:
                    yield GenerationWorkItem(
                        session=session,
                        candidate_index=candidate_index,
                    )


def _require_trial_execution_session(
    session: TrialExecutionSession,
) -> TrialExecutionSession:
    if not isinstance(session, TrialExecutionSession):
        raise TypeError(
            "Execution runner must return a TrialExecutionSession from "
            "prepare_trial_session()."
        )
    return session
