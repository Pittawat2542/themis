"""Tests for package metadata and exports."""

import themis


def test_version_string_format():
    assert isinstance(themis.__version__, str)
    assert themis.__version__
    assert "." in themis.__version__


def test_package_exports_accessible():
    """Lazy-loaded submodules should be accessible via attribute access."""
    assert themis.config is not None
    assert themis.evaluation is not None
    assert themis.generation is not None


def test_core_api_accessible():
    """Primary API functions should be importable directly."""
    assert callable(themis.evaluate)
    assert callable(themis.register_metric)
    assert callable(themis.register_provider)
    assert callable(themis.register_dataset)
    assert callable(themis.register_benchmark)
    assert callable(themis.list_providers)
    assert callable(themis.list_datasets)
    assert callable(themis.list_benchmarks)


def test_exceptions_accessible():
    """Exception classes should be directly accessible."""
    assert issubclass(themis.ThemisError, Exception)
    assert issubclass(themis.ConfigurationError, themis.ThemisError)
    assert issubclass(themis.ProviderError, themis.ThemisError)
