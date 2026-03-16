from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import logging

from themis.progress.models import RunProgressSnapshot

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ProgressEventType(StrEnum):
    """Known progress lifecycle events emitted by Themis orchestration."""

    RUN_STARTED = "run_started"
    STAGE_STARTED = "stage_started"
    WORK_ITEM_STARTED = "work_item_started"
    WORK_ITEM_FINISHED = "work_item_finished"
    RUN_FINISHED = "run_finished"


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable progress event emitted to progress subscribers."""

    event_type: ProgressEventType
    snapshot: RunProgressSnapshot
    emitted_at: datetime = field(default_factory=_now_utc)


ProgressSubscriber = Callable[[ProgressEvent], None]


class ProgressBus:
    """Minimal in-process pub/sub bus for progress snapshots."""

    def __init__(self, *, fail_fast_subscribers: bool = False) -> None:
        self._subscribers: list[ProgressSubscriber] = []
        self.fail_fast_subscribers = fail_fast_subscribers

    def subscribe(self, subscriber: ProgressSubscriber) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: ProgressSubscriber) -> None:
        self._subscribers = [
            existing for existing in self._subscribers if existing is not subscriber
        ]

    def emit(
        self,
        event_type: ProgressEventType,
        *,
        snapshot: RunProgressSnapshot,
    ) -> ProgressEvent:
        event = ProgressEvent(event_type=event_type, snapshot=snapshot)
        for subscriber in tuple(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                if self.fail_fast_subscribers:
                    raise
                logger.exception("Progress subscriber failed for %s", event.event_type)
        return event
