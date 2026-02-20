"""Tests for the new configuration system."""

from pathlib import Path

import pytest

from themis.config import registry, runtime, schema
from themis.experiment import orchestrator


def test_experiment_config_from_dict():
    """Test creating ExperimentConfig from a dictionary."""
    data = {
        "name": "test_experiment",
        "task": "math500",
        "dataset": {"source": "huggingface", "split": "test", "dataset_id": "math500"},
        "generation": {
            "model_identifier": "test-model",
            "provider": {"name": "fake"},
        },
    }
    config = schema.ExperimentConfig.from_dict(data)
    assert config.name == "test_experiment"
    assert config.task == "math500"
    assert config.dataset.source == "huggingface"
    assert config.dataset.dataset_id == "math500"
    assert config.generation.model_identifier == "test-model"


def test_experiment_config_io(tmp_path: Path):
    """Test saving and loading ExperimentConfig to/from a file."""
    config = schema.ExperimentConfig(
        name="test_io",
        task="math500",
        dataset=schema.DatasetConfig(source="inline"),
    )
    file_path = tmp_path / "config.yaml"
    config.to_file(file_path)

    assert file_path.exists()

    loaded_config = schema.ExperimentConfig.from_file(file_path)
    assert loaded_config.name == "test_io"
    assert loaded_config.task == "math500"
    assert loaded_config.dataset.source == "inline"


def test_custom_task_registration():
    """Test registering and using a custom task."""

    @registry.register_experiment_builder("custom_task")
    def build_custom_experiment(
        config: schema.ExperimentConfig,
    ) -> orchestrator.ExperimentOrchestrator:
        # Return a dummy orchestrator (mocking for simplicity)
        # In a real test we might want to return a real object or mock
        return "custom_orchestrator"  # type: ignore

    config = schema.ExperimentConfig(
        name="custom_run",
        task="custom_task",
    )

    builder = registry.get_experiment_builder("custom_task")
    assert builder is not None

    result = runtime._build_experiment(config)
    assert result == "custom_orchestrator"


def test_missing_task_raises_error():
    """Test that missing task raises ValueError."""
    config = schema.ExperimentConfig(
        name="test_missing_task",
        # task is None
        dataset=schema.DatasetConfig(source="huggingface", dataset_id="math500"),
    )

    with pytest.raises(
        ValueError, match="Experiment configuration must specify a 'task'"
    ):
        runtime._build_experiment(config)


def test_missing_dataset_id_raises_error():
    """Test that missing dataset_id raises ValueError for non-inline source."""
    config = schema.ExperimentConfig(
        name="test_missing_dataset",
        task="math500",
        dataset=schema.DatasetConfig(source="huggingface"),  # dataset_id is None
    )

    # We need to mock _build_experiment to call _load_dataset or test _load_dataset directly
    # But runtime.run_experiment_from_config calls both.
    # Let's test _load_dataset directly as it's easier.

    with pytest.raises(ValueError, match="dataset.dataset_id must be provided"):
        runtime._load_dataset(config.dataset, experiment_name=config.name)


def test_task_override_via_options():
    """Test overriding task name via task_options."""
    config = schema.ExperimentConfig(
        name="math500_zero_shot",
        task="math500",
        task_options={"task_name": "aime24"},
        dataset=schema.DatasetConfig(source="huggingface", dataset_id="math500"),
        generation=schema.GenerationConfig(
            model_identifier="test", provider=schema.ProviderConfig(name="fake")
        ),
    )

    orchestrator_obj = runtime._build_experiment(config)
    assert orchestrator_obj._plan.templates[0].metadata["task"] == "aime24"
