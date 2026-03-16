"""Langfuse-backed telemetry subscriber for persisted Themis runs."""

from __future__ import annotations

from typing import Protocol, cast

from themis.contracts.protocols import ObservabilityStore
from themis._optional import import_optional
from themis.overlays import OverlaySelection
from themis.records.observability import ObservabilityLink
from themis.telemetry.bus import TelemetryBus, TelemetryEvent, TelemetryEventName
from themis.types.json_types import JSONDict


class _LangfuseTrace(Protocol):
    """Minimal trace handle returned by the Langfuse client."""

    id: str


class _LangfuseClient(Protocol):
    """Subset of the Langfuse client used by the callback."""

    base_url: str | None

    def trace(
        self,
        *,
        name: str | None = None,
        input: JSONDict | None = None,
        id: str | None = None,
        output: JSONDict | None = None,
    ) -> _LangfuseTrace:
        """Create or update a trace."""
        ...

    def span(
        self,
        *,
        trace_id: str,
        name: str,
        input: JSONDict | None = None,
    ) -> object:
        """Create one child span under the current trace."""
        ...


class LangfuseCallback:
    """Telemetry bus subscriber that forwards trial events to Langfuse."""

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        client: _LangfuseClient | None = None,
        observability_store: ObservabilityStore | None = None,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
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
            self.client = cast(_LangfuseClient, client_class(**kwargs))
        else:
            self.client = client

        self._trial_trace_ids: dict[str, str] = {}
        self.observability_store = observability_store
        self.overlay_selection = OverlaySelection(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        resolved_base_url = base_url or getattr(self.client, "base_url", None)
        self.base_url = (
            resolved_base_url if isinstance(resolved_base_url, str) else None
        )

    def subscribe(self, bus: TelemetryBus) -> None:
        """Subscribe this callback to one telemetry bus.

        Args:
            bus: Telemetry bus that should forward events to this callback.

        Returns:
            None.
        """
        bus.subscribe(self.on_event)

    def on_event(self, event: TelemetryEvent) -> None:
        """Handle one telemetry event from the subscribed runtime bus.

        Args:
            event: Telemetry event emitted by the Themis runtime.

        Returns:
            None.
        """
        trial_hash = self._trial_hash(event)
        if trial_hash is None:
            return

        if event.name == TelemetryEventName.TRIAL_START:
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

        if event.name == TelemetryEventName.TRIAL_END:
            self.client.trace(id=trace_id, output=event.payload)
            self._persist_refs(trial_hash, None, trace_id)
            return

        self.client.span(trace_id=trace_id, name=str(event.name), input=event.payload)

    def _trial_hash(self, event: TelemetryEvent) -> str | None:
        trial_hash = event.payload.get("trial_hash")
        return trial_hash if isinstance(trial_hash, str) else None

    def _persist_refs(
        self, trial_hash: str, candidate_id: str | None, trace_id: str
    ) -> None:
        if self.observability_store is None:
            return
        self.observability_store.save_link(
            trial_hash,
            candidate_id,
            self.overlay_selection.overlay_key,
            ObservabilityLink(
                provider="langfuse",
                external_id=trace_id,
                url=self._trace_url(trace_id),
            ),
        )

    def _trace_url(self, trace_id: str) -> str | None:
        if not isinstance(self.base_url, str) or not self.base_url:
            return None
        return f"{self.base_url.rstrip('/')}/trace/{trace_id}"
