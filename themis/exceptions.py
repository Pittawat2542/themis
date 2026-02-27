"""Themis exception hierarchy.

All Themis-specific exceptions inherit from ThemisError, allowing
users to catch any Themis error with a single except clause:

    try:
        report = themis.evaluate(...)
    except themis.ThemisError as e:
        handle_gracefully(e)

Each domain exception also inherits from its stdlib counterpart so
existing ``except ValueError:`` / ``except KeyError:`` handlers
continue to work — this is strictly additive, not a breaking change.
"""

from __future__ import annotations


class ThemisError(Exception):
    """Base exception for all Themis errors."""


class ConfigurationError(ThemisError, ValueError):
    """Invalid configuration, parameters, or options."""


class ProviderError(ThemisError, KeyError):
    """Model provider not found or provider failure."""


class DatasetError(ThemisError, ValueError):
    """Dataset loading, validation, or registry failure."""


class MetricError(ThemisError, ValueError):
    """Metric resolution or computation failure."""


class EvaluationError(ThemisError, RuntimeError):
    """Evaluation pipeline failure."""


class StorageError(ThemisError, RuntimeError):
    """Storage backend failure."""


class DependencyError(ThemisError, ImportError):
    """Missing optional dependency."""


__all__ = [
    "ThemisError",
    "ConfigurationError",
    "ProviderError",
    "DatasetError",
    "MetricError",
    "EvaluationError",
    "StorageError",
    "DependencyError",
]
