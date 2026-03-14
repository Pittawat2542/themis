"""Canonical Themis exception hierarchy used by the v2 runtime."""

from __future__ import annotations

from themis.types.enums import ErrorCode
from themis.types.json_types import JSONValueType


class ThemisError(Exception):
    """
    Base exception for all Themis errors.
    All subclasses carry stable `code`, human-readable `message`, and structured `details`.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, JSONValueType] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class SpecValidationError(ThemisError):
    """Raised when semantic validation of a Spec fails or plugin version is incompatible."""

    pass


class InferenceError(ThemisError):
    """Raised when a model provider fails."""

    pass


class ExtractionError(ThemisError):
    """Raised when an extractor fails to parse model output."""

    pass


class MetricError(ThemisError):
    """Raised when a custom metric computation fails."""

    pass


class StorageError(ThemisError):
    """Raised on storage read/write failures."""

    pass


class OrchestrationAbortedError(ThemisError):
    """Raised when the circuit breaker threshold is triggered, halting the execution matrix."""

    pass


class RetryableProviderError(InferenceError):
    """Specific InferenceError that can be safely retried."""

    pass
