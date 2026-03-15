"""Terminal trial finalization helpers for generation-stage execution."""

from __future__ import annotations

from collections.abc import Callable

from themis.orchestration.runner_state import TrialExecutionSession
from themis.records.candidate import CandidateRecord
from themis.records.trial import TrialRecord
from themis.types.enums import RecordStatus
from themis.types.events import TrialEventType


class TrialFinalizer:
    """Builds terminal generation trial records and emits the final trial event."""

    def __init__(self, *, append_session_event: Callable[..., None]) -> None:
        self.append_session_event = append_session_event

    def finalize_generation_trial(
        self,
        session: TrialExecutionSession,
        candidates: list[CandidateRecord],
    ) -> TrialRecord:
        """Emit the terminal trial event and return the generation trial record."""
        overall_status = RecordStatus.OK
        trial_error = None
        for candidate in sorted(candidates, key=lambda item: item.sample_index):
            if candidate.status == RecordStatus.ERROR:
                overall_status = RecordStatus.ERROR
                trial_error = candidate.error
                break

        trial_record = TrialRecord(
            spec_hash=session.trial_hash,
            trial_spec=session.trial,
            status=overall_status,
            error=trial_error,
            candidates=sorted(candidates, key=lambda item: item.sample_index),
            provenance=session.provenance,
        )

        if overall_status == RecordStatus.ERROR:
            self.append_session_event(
                session,
                TrialEventType.TRIAL_FAILED,
                status=overall_status,
                payload={"status": overall_status.value},
                error=trial_error,
            )
        else:
            self.append_session_event(
                session,
                TrialEventType.TRIAL_COMPLETED,
                status=overall_status,
                payload={"status": overall_status.value},
            )
        return trial_record
