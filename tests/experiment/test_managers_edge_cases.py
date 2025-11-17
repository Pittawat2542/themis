"""Simplified edge case tests for Phase 2 managers.

This module tests critical edge cases for CacheManager and IntegrationManager
without requiring complex entity construction.
"""

import tempfile

import pytest

from themis.config.schema import HuggingFaceHubConfig, IntegrationsConfig, WandbConfig
from themis.experiment.cache_manager import CacheManager
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.storage import ExperimentStorage


# ==== CacheManager Edge Cases ====


def test_cache_manager_with_no_storage():
    """Test CacheManager behaves correctly when storage is None."""
    manager = CacheManager(storage=None, enable_resume=True, enable_cache=True)

    # Should indicate no storage
    assert not manager.has_storage

    # All operations should be no-ops without errors
    manager.cache_dataset("test-run", [{"id": "1"}])

    records = manager.load_cached_records("test-run")
    assert records == {}

    evaluations = manager.load_cached_evaluations("test-run")
    assert evaluations == {}

    # Should return None for run path
    run_path = manager.get_run_path("test-run")
    assert run_path is None


def test_cache_manager_with_resume_disabled():
    """Test CacheManager doesn't load cache when resume is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ExperimentStorage(tmpdir)

        # Create manager with resume disabled
        manager = CacheManager(storage=storage, enable_resume=False, enable_cache=True)

        # Should still have storage
        assert manager.has_storage

        # But loading should return empty dicts
        records = manager.load_cached_records("test-run")
        assert records == {}

        evaluations = manager.load_cached_evaluations("test-run")
        assert evaluations == {}


def test_cache_manager_with_empty_cache():
    """Test loading from empty cache returns empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ExperimentStorage(tmpdir)
        manager = CacheManager(storage=storage, enable_resume=True, enable_cache=True)

        # Load from non-existent run
        records = manager.load_cached_records("nonexistent-run")
        assert records == {}

        evaluations = manager.load_cached_evaluations("nonexistent-run")
        assert evaluations == {}


def test_cache_manager_cache_empty_dataset():
    """Test caching empty dataset doesn't cause errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ExperimentStorage(tmpdir)
        manager = CacheManager(storage=storage, enable_resume=True, enable_cache=True)

        # Cache empty dataset - should not raise
        manager.cache_dataset("test-run", [])


def test_cache_manager_get_run_path_returns_string():
    """Test get_run_path returns correct type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ExperimentStorage(tmpdir)
        manager = CacheManager(storage=storage, enable_resume=True, enable_cache=True)

        run_path = manager.get_run_path("test-run-123")

        # Should return path as string
        assert isinstance(run_path, str)
        assert "test-run-123" in run_path


def test_cache_manager_with_both_flags_disabled():
    """Test CacheManager with both resume and cache disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ExperimentStorage(tmpdir)

        # Create manager with everything disabled
        manager = CacheManager(
            storage=storage, enable_resume=False, enable_cache=False
        )

        # Should still have storage
        assert manager.has_storage

        # But loading should return empty
        loaded = manager.load_cached_records("test-run")
        assert loaded == {}


# ==== IntegrationManager Edge Cases ====


def test_integration_manager_with_no_config():
    """Test IntegrationManager with None config uses defaults."""
    manager = IntegrationManager(config=None)

    # Should create with default (empty) config
    assert not manager.has_wandb
    assert not manager.has_huggingface

    # Operations should be no-ops without errors
    manager.initialize_run({"run_id": "test"})
    manager.log_results(None)  # type: ignore
    manager.upload_results(None, None)  # type: ignore
    manager.finalize()


def test_integration_manager_with_empty_config():
    """Test IntegrationManager with empty IntegrationsConfig."""
    config = IntegrationsConfig()
    manager = IntegrationManager(config=config)

    # Should have no integrations enabled
    assert not manager.has_wandb
    assert not manager.has_huggingface


def test_integration_manager_with_wandb_disabled():
    """Test IntegrationManager with WandB explicitly disabled."""
    config = IntegrationsConfig(
        wandb=WandbConfig(enable=False),
    )
    manager = IntegrationManager(config=config)

    # WandB should be disabled
    assert not manager.has_wandb

    # Operations should be no-ops
    manager.initialize_run({"run_id": "test"})
    manager.finalize()


def test_integration_manager_with_huggingface_disabled():
    """Test IntegrationManager with HuggingFace explicitly disabled."""
    config = IntegrationsConfig(
        huggingface_hub=HuggingFaceHubConfig(enable=False),
    )
    manager = IntegrationManager(config=config)

    # HuggingFace should be disabled
    assert not manager.has_huggingface

    # Operations should be no-ops
    manager.upload_results(None, None)  # type: ignore
    manager.finalize()


def test_integration_manager_with_both_disabled():
    """Test IntegrationManager with all integrations disabled."""
    config = IntegrationsConfig(
        wandb=WandbConfig(enable=False),
        huggingface_hub=HuggingFaceHubConfig(enable=False),
    )
    manager = IntegrationManager(config=config)

    # Both should be disabled
    assert not manager.has_wandb
    assert not manager.has_huggingface

    # All operations should be safe no-ops
    manager.initialize_run({"run_id": "test", "max_samples": 10})
    manager.log_results(None)  # type: ignore
    manager.upload_results(None, None)  # type: ignore
    manager.finalize()


def test_integration_manager_initialize_with_empty_dict():
    """Test initialize_run with empty config dictionary."""
    manager = IntegrationManager(config=None)

    # Should handle empty dict without errors
    manager.initialize_run({})


def test_integration_manager_initialize_with_various_configs():
    """Test initialize_run with different configuration dictionaries."""
    manager = IntegrationManager(config=None)

    # Test with various config combinations
    configs = [
        {"run_id": "test-1"},
        {"max_samples": 100},
        {"resume": True},
        {"run_id": "test-2", "max_samples": 50, "resume": False},
        {},  # Empty
    ]

    for config_dict in configs:
        # Should handle all configs without errors
        manager.initialize_run(config_dict)


def test_integration_manager_finalize_multiple_times():
    """Test finalize can be called multiple times safely."""
    manager = IntegrationManager(config=None)

    # Should be safe to call multiple times
    manager.finalize()
    manager.finalize()
    manager.finalize()


def test_integration_manager_methods_in_any_order():
    """Test methods can be called in any order without errors."""
    manager = IntegrationManager(config=None)

    # Call in unusual order
    manager.finalize()
    manager.upload_results(None, None)  # type: ignore
    manager.log_results(None)  # type: ignore
    manager.initialize_run({})
    manager.finalize()


def test_integration_manager_property_checks():
    """Test property check methods work correctly."""
    # All disabled
    manager1 = IntegrationManager(config=None)
    assert not manager1.has_wandb
    assert not manager1.has_huggingface

    # Explicitly disabled
    config2 = IntegrationsConfig(
        wandb=WandbConfig(enable=False),
        huggingface_hub=HuggingFaceHubConfig(enable=False),
    )
    manager2 = IntegrationManager(config=config2)
    assert not manager2.has_wandb
    assert not manager2.has_huggingface
