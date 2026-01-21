"""Tests for storage improvements: compression, deduplication, configuration."""

import gzip
import json
from pathlib import Path

import pytest

from themis.core import entities as core_entities
from themis.experiment.storage import (
    STORAGE_FORMAT_VERSION,
    ExperimentStorage,
    StorageConfig,
    task_cache_key,
)


@pytest.fixture
def sample_task():
    """Create a sample generation task."""
    return core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=core_entities.PromptSpec(
                name="test-prompt",
                template="Answer: {question}",
                metadata={"format": "simple"},
            ),
            text="Answer: What is 2+2?",
            context={"question": "What is 2+2?"},
        ),
        model=core_entities.ModelSpec(
            identifier="test-model",
            provider="test",
        ),
        sampling=core_entities.SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        ),
        metadata={"sample_id": "test-001"},
        reference=core_entities.Reference(kind="answer", value="4"),
    )


@pytest.fixture
def sample_record(sample_task):
    """Create a sample generation record."""
    return core_entities.GenerationRecord(
        task=sample_task,
        output=core_entities.ModelOutput(
            text="The answer is 4",
            raw={"id": "test", "choices": [{"message": {"content": "The answer is 4"}}]},
        ),
        error=None,
        metrics={"cost_usd": 0.001},
    )


def test_storage_config_defaults():
    """Test StorageConfig default values."""
    config = StorageConfig()
    assert config.save_raw_responses is False
    assert config.save_dataset is True
    assert config.compression == "gzip"
    assert config.format_version == STORAGE_FORMAT_VERSION
    assert config.deduplicate_templates is True


def test_storage_config_custom():
    """Test StorageConfig with custom values."""
    config = StorageConfig(
        save_raw_responses=True,
        compression="none",
        deduplicate_templates=False,
    )
    assert config.save_raw_responses is True
    assert config.compression == "none"
    assert config.deduplicate_templates is False


def test_storage_with_compression(tmp_path, sample_record):
    """Test storage with gzip compression enabled."""
    config = StorageConfig(compression="gzip")
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    # Append record
    storage.append_record(run_id, sample_record)

    # Check that .gz file exists
    records_path = tmp_path / run_id / "records.jsonl.gz"
    assert records_path.exists()

    # Verify it's gzipped
    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 2  # Header + record
        # First line should be header
        header = json.loads(lines[0])
        assert header["_type"] == "header"
        assert header["_format_version"] == STORAGE_FORMAT_VERSION


def test_storage_without_compression(tmp_path, sample_record):
    """Test storage without compression."""
    config = StorageConfig(compression="none")
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    storage.append_record(run_id, sample_record)

    # Check that regular file exists (not .gz)
    records_path = tmp_path / run_id / "records.jsonl"
    assert records_path.exists()

    with records_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 2


def test_storage_without_raw_responses(tmp_path, sample_record):
    """Test that raw responses are not saved when disabled."""
    config = StorageConfig(save_raw_responses=False)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    storage.append_record(run_id, sample_record)

    # Load and verify no raw field
    records = storage.load_cached_records(run_id)
    assert len(records) == 1
    record = next(iter(records.values()))
    assert record.output.raw is None


def test_storage_with_raw_responses(tmp_path, sample_record):
    """Test that raw responses are saved when enabled."""
    config = StorageConfig(save_raw_responses=True)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    storage.append_record(run_id, sample_record)

    # Load and verify raw field exists
    records = storage.load_cached_records(run_id)
    assert len(records) == 1
    record = next(iter(records.values()))
    assert record.output.raw is not None
    assert record.output.raw["id"] == "test"


def test_template_deduplication(tmp_path, sample_task):
    """Test template deduplication functionality."""
    config = StorageConfig(deduplicate_templates=True)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    # Create multiple tasks with same template
    task1 = sample_task
    task2 = core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=sample_task.prompt.spec,  # Same spec
            text="Answer: What is 3+3?",
            context={"question": "What is 3+3?"},
        ),
        model=task1.model,
        sampling=task1.sampling,
        metadata={"sample_id": "test-002"},
        reference=core_entities.Reference(kind="answer", value="6"),
    )

    record1 = core_entities.GenerationRecord(
        task=task1,
        output=core_entities.ModelOutput(text="4"),
        error=None,
    )
    record2 = core_entities.GenerationRecord(
        task=task2,
        output=core_entities.ModelOutput(text="6"),
        error=None,
    )

    storage.append_record(run_id, record1)
    storage.append_record(run_id, record2)

    # Check templates file exists
    templates_path = tmp_path / run_id / "templates.jsonl.gz"
    assert templates_path.exists()

    # Load templates and verify only one template stored
    templates = storage._load_templates(run_id)
    assert len(templates) == 1  # Only one unique template

    # Load tasks and verify they're restored correctly
    tasks = storage._load_tasks(run_id)
    assert len(tasks) == 2
    for task in tasks.values():
        assert task.prompt.spec.template == sample_task.prompt.spec.template


def test_no_template_deduplication(tmp_path, sample_task):
    """Test storage without template deduplication."""
    config = StorageConfig(deduplicate_templates=False)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    record = core_entities.GenerationRecord(
        task=sample_task,
        output=core_entities.ModelOutput(text="4"),
        error=None,
    )

    storage.append_record(run_id, record)

    # Templates file should not be created
    templates_path = tmp_path / run_id / "templates.jsonl"
    templates_path_gz = tmp_path / run_id / "templates.jsonl.gz"
    assert not templates_path.exists()
    assert not templates_path_gz.exists()


def test_format_versioning(tmp_path, sample_record):
    """Test that format version is written to files."""
    storage = ExperimentStorage(tmp_path)
    run_id = "test-run"

    storage.append_record(run_id, sample_record)

    # Read records file and check header
    records_path = tmp_path / run_id / "records.jsonl.gz"
    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        header = json.loads(first_line)
        assert header["_type"] == "header"
        assert header["_format_version"] == STORAGE_FORMAT_VERSION
        assert header["_file_type"] == "records"


def test_dataset_saving_disabled(tmp_path):
    """Test that dataset is not saved when disabled."""
    config = StorageConfig(save_dataset=False)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    dataset = [{"id": "1", "text": "sample"}]
    storage.cache_dataset(run_id, dataset)

    # Dataset file should not exist
    dataset_path = tmp_path / run_id / "dataset.jsonl"
    dataset_path_gz = tmp_path / run_id / "dataset.jsonl.gz"
    assert not dataset_path.exists()
    assert not dataset_path_gz.exists()


def test_dataset_saving_enabled(tmp_path):
    """Test that dataset is saved when enabled."""
    config = StorageConfig(save_dataset=True)
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    dataset = [{"id": "1", "text": "sample"}]
    storage.cache_dataset(run_id, dataset)

    # Dataset file should exist
    dataset_path_gz = tmp_path / run_id / "dataset.jsonl.gz"
    assert dataset_path_gz.exists()

    # Load and verify
    loaded = storage.load_dataset(run_id)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "1"


def test_round_trip_with_all_optimizations(tmp_path, sample_task):
    """Test complete round trip with all optimizations enabled."""
    config = StorageConfig(
        save_raw_responses=False,
        compression="gzip",
        deduplicate_templates=True,
    )
    storage = ExperimentStorage(tmp_path, config=config)
    run_id = "test-run"

    # Create and save multiple records with same template
    records_to_save = []
    for i in range(5):
        task = core_entities.GenerationTask(
            prompt=core_entities.PromptRender(
                spec=sample_task.prompt.spec,
                text=f"Answer: What is {i}+{i}?",
                context={"question": f"What is {i}+{i}?"},
            ),
            model=sample_task.model,
            sampling=sample_task.sampling,
            metadata={"sample_id": f"test-{i:03d}"},
            reference=core_entities.Reference(kind="answer", value=str(i * 2)),
        )
        record = core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=str(i * 2)),
            error=None,
        )
        records_to_save.append(record)
        storage.append_record(run_id, record)

    # Load and verify
    loaded_records = storage.load_cached_records(run_id)
    assert len(loaded_records) == 5

    # Verify data integrity
    for original in records_to_save:
        key = task_cache_key(original.task)
        loaded = loaded_records[key]
        assert loaded.task.prompt.spec.template == original.task.prompt.spec.template
        assert loaded.output.text == original.output.text
        assert loaded.output.raw is None  # Raw responses disabled
        assert loaded.task.metadata["sample_id"] == original.task.metadata["sample_id"]

    # Verify storage efficiency
    templates = storage._load_templates(run_id)
    assert len(templates) == 1  # Only one template despite 5 tasks
