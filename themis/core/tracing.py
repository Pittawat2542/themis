"""Tracing helpers for runtime execution."""

from __future__ import annotations


class NoOpTracingProvider:
    def start_span(self, name: str, attributes: dict[str, object]) -> object:
        del name, attributes
        return "noop"

    def end_span(self, span: object, status: str) -> None:
        del span, status
