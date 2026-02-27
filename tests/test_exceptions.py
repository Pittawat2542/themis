"""Tests for the Themis exception hierarchy."""

import pytest

from themis.exceptions import (
    ThemisError,
    ConfigurationError,
    ProviderError,
    DatasetError,
    MetricError,
    EvaluationError,
    StorageError,
)


class TestExceptionHierarchy:
    """All domain exceptions inherit from ThemisError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ProviderError,
            DatasetError,
            MetricError,
            EvaluationError,
            StorageError,
        ],
    )
    def test_inherits_from_themis_error(self, exc_cls):
        assert issubclass(exc_cls, ThemisError)

    def test_catch_all_with_themis_error(self):
        """Users can catch any Themis error with a single except clause."""
        for exc_cls in (
            ConfigurationError,
            ProviderError,
            DatasetError,
            MetricError,
            EvaluationError,
            StorageError,
        ):
            with pytest.raises(ThemisError):
                raise exc_cls("test")


class TestBackwardCompatibility:
    """Domain exceptions are also catchable by stdlib types."""

    def test_configuration_error_is_value_error(self):
        with pytest.raises(ValueError):
            raise ConfigurationError("bad config")

    def test_provider_error_is_key_error(self):
        with pytest.raises(KeyError):
            raise ProviderError("missing provider")

    def test_dataset_error_is_value_error(self):
        with pytest.raises(ValueError):
            raise DatasetError("bad dataset")

    def test_metric_error_is_value_error(self):
        with pytest.raises(ValueError):
            raise MetricError("bad metric")

    def test_evaluation_error_is_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise EvaluationError("eval failed")

    def test_storage_error_is_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise StorageError("storage failed")


class TestExceptionMessages:
    """Exceptions preserve their messages."""

    def test_message_preserved(self):
        msg = "Something went wrong with configuration"
        err = ConfigurationError(msg)
        assert str(err) == msg

    def test_themis_error_is_exception(self):
        assert issubclass(ThemisError, Exception)
        assert not issubclass(ThemisError, (ValueError, KeyError, RuntimeError))
