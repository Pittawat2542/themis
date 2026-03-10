from __future__ import annotations

from typing import Any

from themis._optional import import_optional
from themis.records.observability import ObservabilityRefs
from themis.storage.observability import SqliteObservabilityStore
from themis.telemetry.bus import TelemetryBus, TelemetryEvent


class LangfuseCallback:
    """Telemetry bus subscriber that forwards trial events to Langfuse."""

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        observability_store: SqliteObservabilityStore | None = None,
        eval_revision: str = "latest",
    ) -> None:
        if client is None:
            module = import_optional("langfuse", extra="telemetry")
            client_class = getattr(module, "Langfuse")
            kwargs = {}
            if public_key is not None:
                kwargs["public_key"] = public_key
            if secret_key is not None:
                kwargs["secret_key"] = secret_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self.client = client_class(**kwargs)
        else:
            self.client = client

        self._trial_trace_ids: dict[str, str] = {}
        self.observability_store = observability_store
        self.eval_revision = eval_revision
        self.base_url = base_url or getattr(self.client, "base_url", None)

    def subscribe(self, bus: TelemetryBus) -> None:
        bus.subscribe(self.on_event)

    def on_event(self, event: TelemetryEvent) -> None:
        trial_hash = self._trial_hash(event)
        if trial_hash is None:
            return

        if event.name == "trial_start":
            trace = self.client.trace(name=trial_hash, input=event.payload)
            self._trial_trace_ids[trial_hash] = trace.id
            self._persist_refs(trial_hash, None, trace.id)
            return

        trace_id = self._trial_trace_ids.get(trial_hash)
        if trace_id is None:
            return

        candidate_id = event.payload.get("candidate_id")
        if isinstance(candidate_id, str):
            self._persist_refs(trial_hash, candidate_id, trace_id)

        if event.name == "trial_end":
            self.client.trace(id=trace_id, output=event.payload)
            self._persist_refs(trial_hash, None, trace_id)
            return

        self.client.span(trace_id=trace_id, name=event.name, input=event.payload)

    def _trial_hash(self, event: TelemetryEvent) -> str | None:
        trial_hash = event.payload.get("trial_hash")
        return trial_hash if isinstance(trial_hash, str) else None

    def _persist_refs(
        self, trial_hash: str, candidate_id: str | None, trace_id: str
    ) -> None:
        if self.observability_store is None:
            return
        self.observability_store.save_refs(
            trial_hash,
            candidate_id,
            self.eval_revision,
            ObservabilityRefs(
                langfuse_trace_id=trace_id,
                langfuse_url=self._trace_url(trace_id),
            ),
        )

    def _trace_url(self, trace_id: str) -> str | None:
        if not isinstance(self.base_url, str) or not self.base_url:
            return None
        return f"{self.base_url.rstrip('/')}/trace/{trace_id}"
