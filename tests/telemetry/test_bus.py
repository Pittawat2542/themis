from __future__ import annotations

import pytest

from themis.errors.exceptions import ThemisError
from themis.records.observability import ObservabilityRefs
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import ErrorCode


def test_telemetry_bus_emits_structured_events_to_subscribers():
    from themis.telemetry.bus import TelemetryBus

    bus = TelemetryBus()
    seen = []
    bus.subscribe(seen.append)

    bus.emit("trial_start", trial_hash="trial-1")
    bus.emit("metric_end", trial_hash="trial-1", candidate_id="cand-1", metric_id="em")

    assert [event.name for event in seen] == ["trial_start", "metric_end"]
    assert seen[0].payload == {"trial_hash": "trial-1"}
    assert seen[1].payload["candidate_id"] == "cand-1"


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

    trial_refs = store.get_refs("trial-1", None, "latest")
    candidate_refs = store.get_refs("trial-1", "cand-1", "latest")

    assert trial_refs == ObservabilityRefs(
        langfuse_trace_id="trace-1",
        langfuse_url="https://langfuse.example/trace/trace-1",
    )
    assert candidate_refs == ObservabilityRefs(
        langfuse_trace_id="trace-1",
        langfuse_url="https://langfuse.example/trace/trace-1",
    )
