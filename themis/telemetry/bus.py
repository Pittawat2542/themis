from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from themis.types.json_types import JSONValueType


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TelemetryEvent:
    """Structured runtime telemetry payload."""

    name: str
    payload: dict[str, JSONValueType] = field(default_factory=dict)
    emitted_at: datetime = field(default_factory=_now_utc)


Subscriber = Callable[[TelemetryEvent], None]


class TelemetryBus:
    """Minimal in-process pub/sub bus for telemetry events."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers = [
            existing_subscriber
            for existing_subscriber in self._subscribers
            if existing_subscriber is not subscriber
        ]

    def emit(self, name: str, **payload: JSONValueType) -> TelemetryEvent:
        event = TelemetryEvent(name=name, payload=dict(payload))
        for subscriber in tuple(self._subscribers):
            subscriber(event)
        return event
