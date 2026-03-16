from __future__ import annotations

import pytest

from themis.errors import ThemisError
from themis.overlays import overlay_key_for
from themis.records.observability import ObservabilityLink, ObservabilitySnapshot
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode


def test_telemetry_bus_emits_structured_events_to_subscribers():
    from themis.telemetry.bus import TelemetryBus, TelemetryEventName

    bus = TelemetryBus()
    seen = []
    bus.subscribe(seen.append)

    bus.emit("trial_start", trial_hash="trial-1")
    bus.emit("metric_end", trial_hash="trial-1", candidate_id="cand-1", metric_id="em")

    assert [event.name for event in seen] == [
        TelemetryEventName.TRIAL_START,
        TelemetryEventName.METRIC_END,
    ]
    assert seen[0].payload == {"trial_hash": "trial-1"}
    assert seen[1].payload["candidate_id"] == "cand-1"


def test_telemetry_bus_isolates_subscriber_failures_by_default(caplog):
    from themis.telemetry.bus import TelemetryBus, TelemetryEventName

    bus = TelemetryBus()
    seen = []

    def explode(event):
        raise RuntimeError("subscriber boom")

    bus.subscribe(explode)
    bus.subscribe(seen.append)

    with caplog.at_level("ERROR"):
        event = bus.emit("trial_end", trial_hash="trial-1", status="ok")

    assert event.name == TelemetryEventName.TRIAL_END
    assert seen == [event]
    assert "Telemetry subscriber failed for trial_end" in caplog.text


def test_telemetry_bus_can_fail_fast_on_subscriber_errors():
    from themis.telemetry.bus import TelemetryBus

    bus = TelemetryBus(fail_fast_subscribers=True)

    def explode(event):
        raise RuntimeError("subscriber boom")

    bus.subscribe(explode)

    with pytest.raises(RuntimeError, match="subscriber boom"):
        bus.emit("trial_start", trial_hash="trial-1")


def test_telemetry_bus_unsubscribe_matches_bound_methods_by_equality():
    from themis.telemetry.bus import TelemetryBus

    class Recorder:
        def __init__(self) -> None:
            self.seen: list[object] = []

        def handle(self, event: object) -> None:
            self.seen.append(event)

    recorder = Recorder()
    bus = TelemetryBus()
    bus.subscribe(recorder.handle)

    bus.unsubscribe(recorder.handle)
    bus.emit("trial_start", trial_hash="trial-1")

    assert recorder.seen == []


def test_langfuse_callback_requires_telemetry_extra(monkeypatch):
    from themis.telemetry import langfuse_callback as module

    def raise_missing_optional(module_name: str, *, extra: str):
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=f'Install it with `uv add "themis-eval[{extra}]"`.',
        )

    monkeypatch.setattr(module, "import_optional", raise_missing_optional)

    with pytest.raises(ThemisError, match=r"themis-eval\[telemetry\]"):
        module.LangfuseCallback(public_key="pk", secret_key="sk")


class MockTrace:
    def __init__(self, trace_id: str):
        self.id = trace_id


class MockLangfuseClient:
    def __init__(self):
        self.traces = []
        self.spans = []

    def trace(self, **kwargs):
        self.traces.append(kwargs)
        return MockTrace("trace-1")

    def span(self, **kwargs):
        self.spans.append(kwargs)


def test_langfuse_callback_persists_observability_refs(tmp_path):
    from themis.telemetry.bus import TelemetryBus
    from themis.telemetry.langfuse_callback import LangfuseCallback

    manager = DatabaseManager(f"sqlite:///{tmp_path}/telemetry.db")
    manager.initialize()
    store = SqliteObservabilityStore(manager)
    callback = LangfuseCallback(
        client=MockLangfuseClient(),
        observability_store=store,
        base_url="https://langfuse.example",
        evaluation_hash="eval-1",
    )
    bus = TelemetryBus()
    callback.subscribe(bus)

    bus.emit("trial_start", trial_hash="trial-1")
    bus.emit(
        "metric_end",
        trial_hash="trial-1",
        candidate_id="cand-1",
        metric_id="em",
        score=1.0,
    )
    bus.emit("trial_end", trial_hash="trial-1", status="ok")

    overlay_key = overlay_key_for(evaluation_hash="eval-1")
    trial_refs = store.get_snapshot("trial-1", None, overlay_key)
    candidate_refs = store.get_snapshot(
        "trial-1",
        "cand-1",
        overlay_key,
    )

    assert trial_refs == ObservabilitySnapshot(
        links=[
            ObservabilityLink(
                provider="langfuse",
                external_id="trace-1",
                url="https://langfuse.example/trace/trace-1",
            )
        ]
    )
    assert candidate_refs == ObservabilitySnapshot(
        links=[
            ObservabilityLink(
                provider="langfuse",
                external_id="trace-1",
                url="https://langfuse.example/trace/trace-1",
            )
        ]
    )
