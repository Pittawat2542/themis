"""Trial execution coordinator with resume checks and circuit breaking."""

import logging
from collections.abc import Mapping, Sequence

from themis.contracts.protocols import (
    DatasetContext,
    ProjectionHandler,
    ProjectionRepository,
    TrialEventRepository,
)
from themis.errors.exceptions import OrchestrationAbortedError
from themis.errors.mapping import map_exception_to_error_record
from themis.orchestration.trial_runner import TrialRunner
from themis.orchestration.trial_planner import PlannedTrial
from themis.records.trial import TrialRecord
from themis.specs.experiment import DataItemContext
from themis.specs.experiment import ExecutionPolicySpec, RuntimeContext, TrialSpec
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import TrialEvent, TrialEventType

logger = logging.getLogger(__name__)


class TrialExecutor:
    """Coordinates trial execution, resume checks, projections, and circuit breaking."""

    def __init__(
        self,
        runner: TrialRunner,
        projection_repo: ProjectionRepository,
        event_repo: TrialEventRepository | None = None,
        projection_handler: ProjectionHandler | None = None,
        execution_policy: ExecutionPolicySpec | None = None,
        telemetry_bus: TelemetryBus | None = None,
    ) -> None:
        self.runner = runner
        self.projection_repo = projection_repo
        self.event_repo = event_repo
        self.projection_handler = projection_handler
        self.execution_policy = execution_policy or ExecutionPolicySpec()
        self._last_failure_fingerprint: str | None = None
        self._consecutive_matching_failures = 0
        self.telemetry_bus = telemetry_bus

    def execute_trials(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        runtime_context: Mapping[str, object] | RuntimeContext | None,
        *,
        dataset_context: DatasetContext | None = None,
        resume: bool = True,
        eval_revision: str = "latest",
    ) -> None:
        """Run trials sequentially, respecting resume and projection semantics."""
        for planned_trial in trials:
            trial_dataset_context: DatasetContext
            if isinstance(planned_trial, PlannedTrial):
                trial = planned_trial.trial_spec
                trial_dataset_context = planned_trial.dataset_context
            else:
                trial = planned_trial
                trial_dataset_context = dataset_context or DataItemContext(
                    item_id=trial.item_id, payload={}
                )

            trial_hash = trial.spec_hash
            if resume and self._should_skip_trial(trial_hash, eval_revision):
                logger.info("Skipping cached trial: %s", trial.trial_id)
                continue

            logger.info("Executing trial: %s", trial.trial_id)
            try:
                trial_record = self.runner.run_trial(
                    trial, trial_dataset_context, runtime_context
                )

                projected_record: TrialRecord | None = None
                if self.projection_handler is not None:
                    projected_record = self.projection_handler.on_trial_completed(
                        trial_hash, eval_revision=eval_revision
                    )
                else:
                    self.projection_repo.save_trial_record(
                        trial_record, eval_revision=eval_revision
                    )

                self._update_circuit_breaker(projected_record or trial_record)
            except Exception as exc:
                self._record_terminal_failure(trial, exc, eval_revision=eval_revision)
                raise

    def _should_skip_trial(self, trial_hash: str, eval_revision: str) -> bool:
        if self.event_repo is not None:
            latest_terminal = self.event_repo.latest_terminal_event_type(trial_hash)
            has_projection = self.event_repo.has_projection_for_revision(
                trial_hash, eval_revision
            )
            if latest_terminal is not None:
                return (
                    latest_terminal == TrialEventType.TRIAL_COMPLETED and has_projection
                )

        return bool(self.projection_repo.has_trial(trial_hash, eval_revision))

    def _update_circuit_breaker(self, trial_record: TrialRecord) -> None:
        if trial_record.status != RecordStatus.ERROR or trial_record.error is None:
            self._last_failure_fingerprint = None
            self._consecutive_matching_failures = 0
            return

        fingerprint = trial_record.error.fingerprint
        if fingerprint == self._last_failure_fingerprint:
            self._consecutive_matching_failures += 1
        else:
            self._last_failure_fingerprint = fingerprint
            self._consecutive_matching_failures = 1

        if (
            self._consecutive_matching_failures
            >= self.execution_policy.circuit_breaker_threshold
        ):
            raise OrchestrationAbortedError(
                code=ErrorCode.CIRCUIT_BREAKER,
                message="Circuit breaker triggered after repeated matching trial failures.",
                details={
                    "fingerprint": fingerprint,
                    "consecutive_failures": self._consecutive_matching_failures,
                    "threshold": self.execution_policy.circuit_breaker_threshold,
                },
            )

    def _record_terminal_failure(
        self,
        trial: TrialSpec,
        exc: Exception,
        *,
        eval_revision: str,
    ) -> None:
        error = map_exception_to_error_record(
            exc,
            provider=trial.model.provider,
            model_id=trial.model.model_id,
            candidate_id=None,
            attempt=None,
        )
        if self.event_repo is not None:
            next_seq = (self.event_repo.last_event_index(trial.spec_hash) or 0) + 1
            try:
                self.event_repo.append_event(
                    TrialEvent(
                        trial_hash=trial.spec_hash,
                        event_seq=next_seq,
                        event_id=f"{trial.spec_hash}:{next_seq}",
                        event_type=TrialEventType.TRIAL_FAILED,
                        status=RecordStatus.ERROR,
                        payload={
                            "status": RecordStatus.ERROR.value,
                            "eval_revision": eval_revision,
                        },
                        error=error,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to append terminal trial_failed event for %s",
                    trial.trial_id,
                )
        if self.telemetry_bus is not None:
            self.telemetry_bus.emit(
                "error",
                trial_hash=trial.spec_hash,
                provider=trial.model.provider,
                model_id=trial.model.model_id,
                code=error.code.value,
                where=error.where.value,
                message=error.message,
            )
            self.telemetry_bus.emit(
                "trial_end",
                trial_hash=trial.spec_hash,
                status=RecordStatus.ERROR.value,
            )
