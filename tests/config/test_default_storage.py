"""Tests for default storage path functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory

from themis.config.schema import ExperimentConfig, StorageConfig
from themis.config.runtime import _build_experiment


def test_default_storage_path_used_when_path_is_none():
    """Test that default_path is used when path is None."""
    with TemporaryDirectory() as tmp_dir:
        default_path = Path(tmp_dir) / "default_storage"

        config = ExperimentConfig(
            task="math500",
            storage=StorageConfig(path=None, default_path=str(default_path))
        )

        # This should not raise an exception
        experiment = _build_experiment(config)

        # Verify that the storage was created with the default path
        assert experiment._storage is not None
        assert experiment._storage._root == default_path


def test_specific_path_takes_precedence_over_default():
    """Test that path takes precedence over default_path when both are specified."""
    with TemporaryDirectory() as tmp_dir:
        specific_path = Path(tmp_dir) / "specific_storage"
        default_path = Path(tmp_dir) / "default_storage"

        config = ExperimentConfig(
            task="math500",
            storage=StorageConfig(
                path=str(specific_path), default_path=str(default_path)
            )
        )

        # This should not raise an exception
        experiment = _build_experiment(config)

        # Verify that the storage was created with the specific path (not the default)
        assert experiment._storage is not None
        assert experiment._storage._root == specific_path


def test_no_storage_when_both_paths_are_none():
    """Test that no storage is created when both path and default_path are None."""
    config = ExperimentConfig(task="math500", storage=StorageConfig(path=None, default_path=None))

    # This should not raise an exception
    experiment = _build_experiment(config)

    # Verify that no storage was created
    assert experiment._storage is None
