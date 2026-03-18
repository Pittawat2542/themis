"""Tests for the Themis exception hierarchy."""

import importlib

import pytest

import themis
from themis.errors import (
    ThemisError,
    ExtractionError,
    InferenceError,
    MetricError,
    OrchestrationAbortedError,
    RetryableProviderError,
    SpecValidationError,
    StorageError,
)
from themis.types.enums import ErrorCode


class TestExceptionHierarchy:
    """All domain exceptions inherit from ThemisError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SpecValidationError,
            InferenceError,
            ExtractionError,
            MetricError,
            StorageError,
            OrchestrationAbortedError,
        ],
    )
    def test_inherits_from_themis_error(self, exc_cls):
        assert issubclass(exc_cls, ThemisError)

    def test_catch_all_with_themis_error(self):
        """Users can catch any Themis error with a single except clause."""
        for exc_cls in (
            SpecValidationError,
            InferenceError,
            ExtractionError,
            MetricError,
            StorageError,
            OrchestrationAbortedError,
        ):
            with pytest.raises(ThemisError):
                raise exc_cls(code=ErrorCode.SCHEMA_MISMATCH, message="test")


class TestExceptionMessages:
    """Exceptions preserve their messages."""

    def test_message_preserved(self):
        msg = "Something went wrong with configuration"
        err = StorageError(code=ErrorCode.STORAGE_READ, message=msg)
        assert str(err) == msg

    def test_themis_error_is_exception(self):
        assert issubclass(ThemisError, Exception)

    def test_root_namespace_keeps_error_types_out_of_curated_surface(self):
        assert not hasattr(themis, "ThemisError")

    def test_retryable_provider_error_is_inference_error(self):
        assert issubclass(RetryableProviderError, InferenceError)

    def test_legacy_exception_module_is_deprecated_shim(self):
        with pytest.deprecated_call():
            legacy_module = importlib.import_module("themis.exceptions")
        assert legacy_module.ThemisError is ThemisError
