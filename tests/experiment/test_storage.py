import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from themis.core import entities as core_entities
from themis.experiment import storage as experiment_storage
from themis.experiment.storage import RetentionPolicy, RunStatus, StorageConfig


def make_record(sample_id: str, answer: str) -> core_entities.GenerationRecord:
    prompt_spec = core_entities.PromptSpec(name="math", template="Solve {problem}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Solve 1+1", context={"problem": "1+1"}, metadata={}
    )
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=32)
    model_spec = core_entities.ModelSpec(identifier="fake", provider="test")
    task = core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": sample_id},
        reference=core_entities.Reference(kind="answer", value=answer),
    )
    return core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text=answer),
        error=None,
        metrics={"latency_ms": 10},
    )


def test_experiment_storage_roundtrip(tmp_path):
    """Test basic storage roundtrip with lifecycle management."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    
    # Start a run
    metadata = storage.start_run("run-1", "exp-1", config={})
    assert metadata.status == RunStatus.IN_PROGRESS
    assert metadata.run_id == "run-1"
    assert metadata.experiment_id == "exp-1"
    
    # Cache dataset
    dataset = [{"id": "1", "problem": "1+1"}]
    storage.cache_dataset("run-1", dataset)

    loaded_dataset = storage.load_dataset("run-1")
    assert loaded_dataset == dataset

    # Append record
    record = make_record("1", "2")
    key = experiment_storage.task_cache_key(record.task)
    storage.append_record("run-1", record, cache_key=key)

    cached = storage.load_cached_records("run-1")
    assert key in cached
    assert cached[key].output.text == "2"

    # Append evaluation without config (backward compatibility)
    score = core_entities.MetricScore(
        metric_name="ExactMatch", value=1.0, details={}, metadata={}
    )
    evaluation = core_entities.EvaluationRecord(sample_id="1", scores=[score])
    storage.append_evaluation("run-1", record, evaluation)

    # Without evaluation_config, should use task key for backward compatibility
    cached_eval = storage.load_cached_evaluations("run-1")
    assert key in cached_eval
    assert cached_eval[key].scores[0].value == 1.0
    
    # Append evaluation with config (new behavior)
    eval_config = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer"
    }
    evaluation2 = core_entities.EvaluationRecord(sample_id="1", scores=[score])
    storage.append_evaluation("run-1", record, evaluation2, evaluation_config=eval_config)
    
    # With evaluation_config, should use evaluation_cache_key
    eval_key = experiment_storage.evaluation_cache_key(record.task, eval_config)
    cached_eval_with_config = storage.load_cached_evaluations("run-1", evaluation_config=eval_config)
    assert eval_key in cached_eval_with_config
    assert cached_eval_with_config[eval_key].scores[0].value == 1.0
    
    # Complete the run
    storage.complete_run("run-1")
    metadata = storage._load_run_metadata("run-1")
    assert metadata.status == RunStatus.COMPLETED


def test_evaluation_cache_invalidation_on_config_change(tmp_path):
    """Test that changing evaluation config invalidates the cache."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    storage.start_run("run-1", "exp-1", config={})
    
    # Create a generation record
    record = make_record("1", "42")
    task_key = experiment_storage.task_cache_key(record.task)
    storage.append_record("run-1", record, cache_key=task_key)
    
    # Save evaluation with config A
    config_a = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer"
    }
    score_a = core_entities.MetricScore(
        metric_name="exact_match", value=1.0, details={}, metadata={}
    )
    eval_a = core_entities.EvaluationRecord(sample_id="1", scores=[score_a])
    storage.append_evaluation("run-1", record, eval_a, evaluation_config=config_a)
    
    # Load with config A - should find it
    eval_key_a = experiment_storage.evaluation_cache_key(record.task, config_a)
    cached_a = storage.load_cached_evaluations("run-1", evaluation_config=config_a)
    assert eval_key_a in cached_a
    assert cached_a[eval_key_a].scores[0].value == 1.0
    
    # Change evaluation config (add new metric)
    config_b = {
        "metrics": ["exact_match", "f1_score"],  # Added new metric
        "extractor": "json_field_extractor:answer"
    }
    
    # Load with config B - should NOT find the cached evaluation
    eval_key_b = experiment_storage.evaluation_cache_key(record.task, config_b)
    cached_b = storage.load_cached_evaluations("run-1", evaluation_config=config_b)
    assert eval_key_b not in cached_b  # Cache miss due to config change
    
    # The keys should be different
    assert eval_key_a != eval_key_b
    
    # Save evaluation with config B
    score_b = core_entities.MetricScore(
        metric_name="f1_score", value=0.8, details={}, metadata={}
    )
    eval_b = core_entities.EvaluationRecord(sample_id="1", scores=[score_a, score_b])
    storage.append_evaluation("run-1", record, eval_b, evaluation_config=config_b)
    
    # Now loading with config B should find it
    cached_b_after = storage.load_cached_evaluations("run-1", evaluation_config=config_b)
    assert eval_key_b in cached_b_after
    assert len(cached_b_after[eval_key_b].scores) == 2
    
    # Loading with config A should still find the old evaluation
    cached_a_still = storage.load_cached_evaluations("run-1", evaluation_config=config_a)
    assert eval_key_a in cached_a_still


def test_storage_config_compression(tmp_path):
    """Test storage with gzip compression."""
    config = StorageConfig(compression="gzip")
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    storage.start_run("run-1", "exp-1", config={})
    dataset = [{"id": str(i), "data": "x" * 100} for i in range(10)]
    storage.cache_dataset("run-1", dataset)
    
    # Check that .gz file was created
    gen_dir = storage._get_generation_dir("run-1")
    assert (gen_dir / "dataset.jsonl.gz").exists()
    assert not (gen_dir / "dataset.jsonl").exists()
    
    # Verify we can load it
    loaded = storage.load_dataset("run-1")
    assert len(loaded) == 10


def test_storage_config_no_raw_responses(tmp_path):
    """Test storage without raw API responses."""
    config = StorageConfig(save_raw_responses=False)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    storage.start_run("run-1", "exp-1", config={})
    
    # Create record with raw data (need to create new record since ModelOutput is frozen)
    record = make_record("1", "2")
    # Create new output with raw field
    output_with_raw = core_entities.ModelOutput(
        text=record.output.text,
        raw={"huge": "data" * 1000},
        usage=record.output.usage,
    )
    # Create new record with the new output
    record_with_raw = core_entities.GenerationRecord(
        task=record.task,
        output=output_with_raw,
        error=record.error,
        metrics=record.metrics,
    )
    
    key = experiment_storage.task_cache_key(record_with_raw.task)
    storage.append_record("run-1", record_with_raw, cache_key=key)
    
    # Verify raw was not saved
    cached = storage.load_cached_records("run-1")
    assert cached[key].output.raw is None


def test_template_deduplication(tmp_path):
    """Test prompt template deduplication."""
    config = StorageConfig(deduplicate_templates=True)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    storage.start_run("run-1", "exp-1", config={})
    
    # Create multiple records with same template
    for i in range(5):
        record = make_record(str(i), str(i))
        storage.append_record("run-1", record)
    
    # Check templates file exists
    gen_dir = storage._get_generation_dir("run-1")
    templates_file = gen_dir / "templates.jsonl"
    assert storage._file_exists_any_compression(templates_file)
    
    # Load and verify only one template stored
    templates = storage._load_templates("run-1")
    assert len(templates) == 1


def test_run_lifecycle(tmp_path):
    """Test run lifecycle management."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    
    # Start run
    metadata = storage.start_run("run-1", "exp-1", config={"model": "gpt-4"})
    assert metadata.status == RunStatus.IN_PROGRESS
    assert metadata.config_snapshot == {"model": "gpt-4"}
    
    # Update progress
    storage.update_run_progress("run-1", total_samples=10, successful_generations=8, failed_generations=2)
    metadata = storage._load_run_metadata("run-1")
    assert metadata.total_samples == 10
    assert metadata.successful_generations == 8
    assert metadata.failed_generations == 2
    
    # Complete run
    storage.complete_run("run-1")
    metadata = storage._load_run_metadata("run-1")
    assert metadata.status == RunStatus.COMPLETED
    assert metadata.completed_at is not None
    
    # Fail run
    storage.start_run("run-2", "exp-1", config={})
    storage.fail_run("run-2", "Test error")
    metadata = storage._load_run_metadata("run-2")
    assert metadata.status == RunStatus.FAILED
    assert metadata.error_message == "Test error"


def test_concurrent_access(tmp_path):
    """Test file locking for concurrent access."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    storage.start_run("run-1", "exp-1", config={})
    
    def append_records(start_id: int):
        for i in range(start_id, start_id + 5):
            record = make_record(str(i), str(i))
            storage.append_record("run-1", record)
    
    # Run concurrent appends
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(append_records, 0),
            executor.submit(append_records, 5),
            executor.submit(append_records, 10),
        ]
        for future in futures:
            future.result()
    
    # Verify all records were saved
    cached = storage.load_cached_records("run-1")
    assert len(cached) == 15


def test_checkpoint_save_and_load(tmp_path):
    """Test checkpoint functionality."""
    config = StorageConfig(checkpoint_interval=5)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    storage.start_run("run-1", "exp-1", config={})
    
    # Append records to trigger checkpoint
    for i in range(10):
        record = make_record(str(i), str(i))
        storage.append_record("run-1", record)
    
    # Load checkpoint
    checkpoint = storage.load_latest_checkpoint("run-1")
    assert checkpoint is not None
    assert checkpoint["total_samples"] >= 5


def test_retention_policy(tmp_path):
    """Test retention policy for cleanup."""
    policy = RetentionPolicy(max_runs_per_experiment=2, keep_latest_n=1)
    config = StorageConfig(retention_policy=policy)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    # Create multiple runs
    for i in range(5):
        run_id = f"run-{i}"
        storage.start_run(run_id, "exp-1", config={})
        storage.complete_run(run_id)
    
    # Apply retention policy
    storage.apply_retention_policy()
    
    # Verify only 2 runs remain
    runs = storage.list_runs(experiment_id="exp-1")
    assert len(runs) <= 2


def test_list_runs_filtering(tmp_path):
    """Test run listing with filters."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    
    # Create runs with different statuses
    storage.start_run("run-1", "exp-1", config={})
    storage.complete_run("run-1")
    
    storage.start_run("run-2", "exp-1", config={})
    storage.fail_run("run-2", "error")
    
    storage.start_run("run-3", "exp-2", config={})
    
    # Test filtering
    all_runs = storage.list_runs()
    assert len(all_runs) == 3
    
    exp1_runs = storage.list_runs(experiment_id="exp-1")
    assert len(exp1_runs) == 2
    
    completed_runs = storage.list_runs(status=RunStatus.COMPLETED)
    assert len(completed_runs) == 1
    assert completed_runs[0].run_id == "run-1"
    
    limited_runs = storage.list_runs(limit=2)
    assert len(limited_runs) == 2


def test_data_integrity_validation(tmp_path):
    """Test data integrity validation."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    storage.start_run("run-1", "exp-1", config={})
    
    record = make_record("1", "2")
    storage.append_record("run-1", record)
    storage.complete_run("run-1")
    
    # Validate integrity
    result = storage.validate_integrity("run-1")
    assert result["valid"]
    assert len(result["errors"]) == 0


def test_storage_size_calculation(tmp_path):
    """Test storage size calculation."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    storage.start_run("run-1", "exp-1", config={})
    
    # Add some data
    dataset = [{"id": str(i), "data": "x" * 1000} for i in range(10)]
    storage.cache_dataset("run-1", dataset)
    
    for i in range(5):
        record = make_record(str(i), str(i))
        storage.append_record("run-1", record)
    
    # Check total storage size (all experiments)
    size = storage.get_storage_size()
    assert size > 0


def test_sqlite_metadata(tmp_path):
    """Test SQLite metadata storage."""
    config = StorageConfig(use_sqlite_metadata=True)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    # Check database was created
    db_path = tmp_path / "experiments.db"
    assert db_path.exists()
    
    # Create a run
    storage.start_run("run-1", "exp-1", config={})
    
    # Query database directly
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT run_id, experiment_id, status FROM runs WHERE run_id = ?", ("run-1",))
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row[0] == "run-1"
    assert row[1] == "exp-1"
    assert row[2] == "in_progress"


def test_format_versioning(tmp_path):
    """Test format versioning in storage files."""
    storage = experiment_storage.ExperimentStorage(tmp_path)
    storage.start_run("run-1", "exp-1", config={})
    
    dataset = [{"id": "1"}]
    storage.cache_dataset("run-1", dataset)
    
    # Read file and check for version header
    gen_dir = storage._get_generation_dir("run-1")
    path = gen_dir / "dataset.jsonl"
    
    with storage._open_for_read(path) as f:
        first_line = f.readline()
        header = json.loads(first_line)
        assert header["_type"] == "header"
        # Check for either 'version' or '_format_version' (both are valid)
        assert "version" in header or "_format_version" in header


def test_resume_from_checkpoint(tmp_path):
    """Test resuming from checkpoint."""
    config = StorageConfig(checkpoint_interval=3)
    storage = experiment_storage.ExperimentStorage(tmp_path, config=config)
    
    storage.start_run("run-1", "exp-1", config={})
    
    # Simulate partial run
    for i in range(5):
        record = make_record(str(i), str(i))
        storage.append_record("run-1", record)
    
    # Load checkpoint to resume
    checkpoint = storage.load_latest_checkpoint("run-1")
    assert checkpoint is not None
    assert checkpoint["total_samples"] >= 3
    
    # Continue from checkpoint
    for i in range(5, 10):
        record = make_record(str(i), str(i))
        storage.append_record("run-1", record)
    
    # Verify all records
    cached = storage.load_cached_records("run-1")
    assert len(cached) == 10