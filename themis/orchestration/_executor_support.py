"""Internal shared execution support for the trial executor."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from themis.contracts.protocols import (
    DatasetContext,
    ProjectionHandler,
    TrialEventRepository,
)
from themis.errors import OrchestrationAbortedError
from themis.errors.mapping import map_exception_to_error_record
from themis.orchestration.trial_planner import PlannedTrial
from themis.records.trial import TrialRecord
from themis.specs.experiment import DataItemContext, ExecutionPolicySpec, TrialSpec
from themis.telemetry.bus import TelemetryBus, TelemetryEventName
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import TrialEvent, TrialEventType

logger = logging.getLogger(__name__)


class ExecutionSupport:
    """Owns shared execution helpers and terminal trial governance."""

    def __init__(
        self,
        *,
        projection_repo,
        event_repo: TrialEventRepository | None,
        projection_handler: ProjectionHandler | None,
        execution_policy: ExecutionPolicySpec,
        telemetry_bus: TelemetryBus | None,
    ) -> None:
        self.projection_repo = projection_repo
        self.event_repo = event_repo
        self.projection_handler = projection_handler
        self.execution_policy = execution_policy
        self.telemetry_bus = telemetry_bus
        self._last_failure_fingerprint: str | None = None
        self._consecutive_matching_failures = 0

    def iter_trials(
        self,
        trials: Sequence[TrialSpec | PlannedTrial],
        *,
        dataset_context: DatasetContext | None,
    ) -> list[tuple[TrialSpec, DatasetContext]]:
        materialized: list[tuple[TrialSpec, DatasetContext]] = []
        for planned_trial in trials:
            if isinstance(planned_trial, PlannedTrial):
                materialized.append(
                    (planned_trial.trial_spec, planned_trial.dataset_context)
                )
                continue
            materialized.append(
                (
                    planned_trial,
                    dataset_context
                    or DataItemContext(item_id=planned_trial.item_id, payload={}),
                )
            )
        return materialized

    def should_skip_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        return bool(
            self.projection_repo.has_trial(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )
        )

    def materialize_projection(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        if self.projection_handler is not None:
            return self.projection_handler.on_trial_completed(
                trial_hash,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )
        return self.projection_repo.materialize_trial_record(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )

    def update_circuit_breaker(self, trial_record: TrialRecord) -> None:
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

    def record_terminal_failure(
        self,
        trial: TrialSpec,
        exc: Exception,
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
                        payload={"status": RecordStatus.ERROR.value},
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
                TelemetryEventName.ERROR,
                trial_hash=trial.spec_hash,
                provider=trial.model.provider,
                model_id=trial.model.model_id,
                code=error.code.value,
                where=error.where.value,
                message=error.message,
            )
            self.telemetry_bus.emit(
                TelemetryEventName.TRIAL_END,
                trial_hash=trial.spec_hash,
                status=RecordStatus.ERROR.value,
            )
