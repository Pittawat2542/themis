"""In-process telemetry bus and event models used by the runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import logging

from themis.types.json_types import JSONValueType

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TelemetryEventName(StrEnum):
    """Known runtime telemetry events emitted by Themis internals."""

    TRIAL_START = "trial_start"
    TRIAL_END = "trial_end"
    CONVERSATION_EVENT = "conversation_event"
    TOOL_CALL = "tool_call"
    EXTRACTOR_ATTEMPT = "extractor_attempt"
    METRIC_START = "metric_start"
    METRIC_END = "metric_end"
    ERROR = "error"


def _normalize_event_name(name: TelemetryEventName | str) -> TelemetryEventName | str:
    if isinstance(name, TelemetryEventName):
        return name
    try:
        return TelemetryEventName(name)
    except ValueError:
        return name


@dataclass(frozen=True)
class TelemetryEvent:
    """Structured runtime telemetry payload."""

    name: TelemetryEventName | str
    payload: dict[str, JSONValueType] = field(default_factory=dict)
    emitted_at: datetime = field(default_factory=_now_utc)


Subscriber = Callable[[TelemetryEvent], None]


class TelemetryBus:
    """Minimal in-process pub/sub bus for telemetry events."""

    def __init__(self, *, fail_fast_subscribers: bool = False) -> None:
        self._subscribers: list[Subscriber] = []
        self.fail_fast_subscribers = fail_fast_subscribers

    def subscribe(self, subscriber: Subscriber) -> None:
        """Register one telemetry subscriber.

        Args:
            subscriber: Callback invoked for every emitted telemetry event.

        Returns:
            None.
        """
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove one telemetry subscriber if it is currently registered.

        Args:
            subscriber: Callback to remove from the subscriber list.

        Returns:
            None.
        """
        self._subscribers = [
            existing_subscriber
            for existing_subscriber in self._subscribers
            if existing_subscriber is not subscriber
        ]

    def emit(
        self,
        name: TelemetryEventName | str,
        **payload: JSONValueType,
    ) -> TelemetryEvent:
        """Emit one telemetry event to the current subscriber set.

        Args:
            name: Event name or enum identifying the telemetry event.
            **payload: JSON-safe event payload forwarded to subscribers.

        Returns:
            The emitted telemetry event object.

        Raises:
            Exception: Re-raised from a subscriber when
                `fail_fast_subscribers=True`.
        """
        event = TelemetryEvent(
            name=_normalize_event_name(name),
            payload=dict(payload),
        )
        for subscriber in tuple(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                if self.fail_fast_subscribers:
                    raise
                logger.exception("Telemetry subscriber failed for %s", event.name)
        return event
