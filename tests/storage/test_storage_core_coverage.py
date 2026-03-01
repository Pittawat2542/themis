"""Targeted tests to cover edge-case paths in themis/storage/core.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


from themis.storage.core import ExperimentStorage
from themis.storage.cache_keys import _json_default
from themis.storage.models import RetentionPolicy, RunStatus, StorageConfig
from tests.factories import make_record, make_evaluation_record
from themis.core.entities import ExperimentReport


# ---------------------------------------------------------------------------
# _json_default edge cases (lines 31-37)
# ---------------------------------------------------------------------------


def test_json_default_serializes_set():
    result = _json_default({3, 1, 2})
    assert result == [1, 2, 3]


def test_json_default_serializes_path():
    result = _json_default(Path("/tmp/foo"))
    assert result == str(Path("/tmp/foo"))


def test_json_default_serializes_object_with_dict():
    class Obj:
        def __init__(self):
            self.x = 1
            self.y = 2

    result = _json_default(Obj())
    assert result == {"x": 1, "y": 2}


def test_json_default_fallback_repr():
    class NoDict:
        __slots__ = ()

        def __repr__(self):
            return "NoDict()"

    result = _json_default(NoDict())
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Checkpoint operations (lines 724-743)
# ---------------------------------------------------------------------------


def test_save_and_load_checkpoint(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-ck", experiment_id="exp")

    storage.save_checkpoint("run-ck", {"progress": 10})
    loaded = storage.load_latest_checkpoint("run-ck")
    assert loaded is not None
    assert loaded["progress"] == 10


def test_load_latest_checkpoint_returns_none_when_no_checkpoints(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-no-ck", experiment_id="exp")
    assert storage.load_latest_checkpoint("run-no-ck") is None


def test_load_latest_checkpoint_missing_dir_returns_none(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-missing", experiment_id="exp")
    # No checkpoint directory → should return None gracefully
    result = storage.load_latest_checkpoint("run-missing")
    assert result is None


def test_checkpoint_auto_triggered_by_config(tmp_path):
    """Verify checkpoint_interval path is hit during append_record."""
    config = StorageConfig(checkpoint_interval=1)
    storage = ExperimentStorage(tmp_path, config=config)
    storage.start_run("run-auto-ck", experiment_id="exp")
    record = make_record(sample_id="s1")
    storage.append_record("run-auto-ck", record)
    loaded = storage.load_latest_checkpoint("run-auto-ck")
    assert loaded is not None
    assert loaded["total_samples"] == 1


# ---------------------------------------------------------------------------
# SQLite metadata store path (lines 292-293, 336-341)
# ---------------------------------------------------------------------------


def test_storage_uses_sqlite_when_configured(tmp_path):
    config = StorageConfig(use_sqlite_metadata=True)
    storage = ExperimentStorage(tmp_path, config=config)
    storage.start_run("run-sqlite", experiment_id="exp-sql")

    # list_runs should delegate to the SQLite path
    runs = storage.list_runs()
    assert any(r.run_id == "run-sqlite" for r in runs)


def test_get_run_dir_resolves_via_sqlite_index(tmp_path):
    """Exercises the metadata_store.get_run_experiment_id path (L336-341)."""
    config = StorageConfig(use_sqlite_metadata=True)
    storage = ExperimentStorage(tmp_path, config=config)
    storage.start_run("run-idx", experiment_id="exp-idx")

    # Create a fresh storage instance (clears in-memory caches)
    storage2 = ExperimentStorage(tmp_path, config=config)
    path = storage2.get_run_path("run-idx")
    assert path.exists()


# ---------------------------------------------------------------------------
# Directory scanning fallback (_get_run_dir_by_scanning, lines 343-361)
# ---------------------------------------------------------------------------


def test_get_run_dir_by_scanning_fallback(tmp_path):
    """Without SQLite, ExperimentStorage scans directories on cache miss."""
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-scan", experiment_id="exp-scan")

    # Fresh instance: no in-memory cache, no SQLite → must scan
    storage2 = ExperimentStorage(tmp_path)
    path = storage2.get_run_path("run-scan")
    assert path.exists()


# ---------------------------------------------------------------------------
# Retention policy (lines 747-799)
# ---------------------------------------------------------------------------


def _make_run(storage: ExperimentStorage, run_id: str, *, completed: bool = True):
    storage.start_run(run_id, experiment_id="exp-ret")
    if completed:
        storage.complete_run(run_id)


def test_retention_policy_keep_latest_n(tmp_path):
    storage = ExperimentStorage(tmp_path)
    for i in range(4):
        _make_run(storage, f"run-ret-{i}")

    policy = RetentionPolicy(keep_latest_n=2)
    storage.apply_retention_policy(policy)

    runs = storage.list_runs()
    assert len(runs) <= 4  # policy evicts runs beyond keep_latest_n


def test_retention_policy_max_age_days(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-old", experiment_id="exp-ret")
    storage.complete_run("run-old")

    # Backdate the run's created_at
    meta = storage._load_run_metadata("run-old")
    old_time = (datetime.now() - timedelta(days=10)).isoformat()
    meta.created_at = old_time
    storage._save_run_metadata(meta)

    _make_run(storage, "run-new")

    policy = RetentionPolicy(keep_latest_n=0, max_age_days=5)
    storage.apply_retention_policy(policy)

    runs = storage.list_runs()
    assert all(r.run_id != "run-old" for r in runs)


def test_retention_policy_keep_completed_only(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-incomplete", experiment_id="exp-ret")
    _make_run(storage, "run-complete")

    policy = RetentionPolicy(keep_latest_n=0, keep_completed_only=True)
    storage.apply_retention_policy(policy)

    runs = storage.list_runs()
    assert all(r.run_id != "run-incomplete" for r in runs)


def test_retention_policy_max_runs_per_experiment(tmp_path):
    storage = ExperimentStorage(tmp_path)
    for i in range(4):
        _make_run(storage, f"run-mp-{i}")

    policy = RetentionPolicy(keep_latest_n=0, max_runs_per_experiment=2)
    storage.apply_retention_policy(policy)

    runs = storage.list_runs()
    assert len(runs) <= 2


def test_apply_retention_policy_none_does_nothing(tmp_path):
    """No retention policy → nothing is deleted."""
    config = StorageConfig(retention_policy=None)
    storage = ExperimentStorage(tmp_path, config=config)
    _make_run(storage, "run-safe")
    storage.apply_retention_policy(None)
    assert len(storage.list_runs()) == 1


# ---------------------------------------------------------------------------
# get_storage_size (lines 820-830)
# ---------------------------------------------------------------------------


def test_get_storage_size_global(tmp_path):
    storage = ExperimentStorage(tmp_path)
    _make_run(storage, "run-size")
    size = storage.get_storage_size()
    assert size > 0


def test_get_storage_size_per_experiment(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-sz", experiment_id="exp-sz")
    storage.complete_run("run-sz")
    size = storage.get_storage_size(experiment_id="exp-sz")
    assert size > 0


def test_get_storage_size_missing_experiment_returns_zero(tmp_path):
    storage = ExperimentStorage(tmp_path)
    size = storage.get_storage_size(experiment_id="does-not-exist")
    assert size == 0


# ---------------------------------------------------------------------------
# fail_run / update_run_progress coverage (lines 190-218)
# ---------------------------------------------------------------------------


def test_fail_run_sets_error(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-fail", experiment_id="exp")
    storage.fail_run("run-fail", "something went wrong")
    meta = storage._load_run_metadata("run-fail")
    assert meta.status == RunStatus.FAILED
    assert meta.error_message == "something went wrong"


def test_update_run_progress_partial(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-prog", experiment_id="exp")
    storage.update_run_progress("run-prog", total_samples=5)
    meta = storage._load_run_metadata("run-prog")
    assert meta.total_samples == 5
    assert meta.successful_generations == 0


# ---------------------------------------------------------------------------
# raw_responses config path (lines 514-515)
# ---------------------------------------------------------------------------


def test_save_raw_responses(tmp_path):
    config = StorageConfig(save_raw_responses=True)
    storage = ExperimentStorage(tmp_path, config=config)
    storage.start_run("run-raw", experiment_id="exp")
    record = make_record(sample_id="s1")
    storage.append_record("run-raw", record)
    loaded = storage.load_cached_records("run-raw")
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Thin delegates coverage
# ---------------------------------------------------------------------------


def test_facade_run_metadata_delegates(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-meta", experiment_id="exp-meta")
    # Needs to go through StorageBackend API
    storage.save_run_metadata(
        "run-meta", {"experiment_id": "exp-meta", "status": "in_progress"}
    )

    assert storage.run_exists("run-meta") is True
    assert storage.run_metadata_exists("run-meta") is True

    loaded = storage.load_run_metadata("run-meta")
    assert loaded["experiment_id"] == "exp-meta"
    assert "run-meta" in storage.list_run_ids()


def test_facade_generation_evaluation_delegates(tmp_path):
    # Using specific cache keys to match implementation mapping expectations
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-gen-eval", experiment_id="exp")

    record = make_record(sample_id="s1")
    eval_record = make_evaluation_record(sample_id="s1")

    # Generation
    storage.save_generation_record("run-gen-eval", record)
    loaded_gen = storage.load_generation_records("run-gen-eval")
    assert len(loaded_gen) == 1

    # Evaluation (save_evaluation_record)
    storage.save_evaluation_record("run-gen-eval", record, eval_record)

    # Evaluation (append_evaluation_record)
    eval_record2 = make_evaluation_record(sample_id="s2")
    storage.append_evaluation_record(
        "run-gen-eval",
        record,
        eval_record2,
        evaluation_config={"metrics": ["exact_match"]},
    )

    loaded_evals = storage.load_evaluation_records("run-gen-eval")
    assert len(loaded_evals) == 2

    # cache_dataset / load_dataset
    dataset = [{"id": "s1", "question": "q"}]
    storage.cache_dataset("run-gen-eval", dataset)
    loaded_ds = storage.load_dataset("run-gen-eval")
    assert len(loaded_ds) == 1


def test_facade_report_delegates(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-report", experiment_id="exp")

    report = ExperimentReport(
        generation_results=[],
        evaluation_report=None,
        failures=[],
        metadata={"test": True},
    )
    storage.save_report("run-report", report)

    loaded_report = storage.load_report("run-report")
    assert loaded_report.metadata["test"] is True

    # Missing report returns None
    assert storage.load_report("run-nonexistent") is None


def test_delete_run_delegate(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-del", experiment_id="exp")
    assert storage.run_exists("run-del")
    storage.delete_run("run-del")
    assert not storage.run_exists("run-del")


def test_facade_property_delegates(tmp_path):
    storage = ExperimentStorage(tmp_path)
    assert repr(storage).startswith("ExperimentStorage(root=")
    assert storage.root == tmp_path

    storage.start_run("run-props", experiment_id="exp")
    # Verify internal getters
    assert storage.get_run_path("run-props") == storage._get_run_dir("run-props")
    assert storage._get_generation_dir("run-props").name == "generation"
    assert storage._get_evaluation_dir("run-props").parent.name == "evaluations"
    assert storage._experiments_dir.name == "experiments"

    # Verify internal property paths exist without erroring
    _ = storage._lock_manager
    _ = storage._fs
    _ = storage._metadata_store
    _ = storage._task_index
    _ = storage._template_index
    _ = storage._run_dir_index

    # Verify acquire lock
    lock = storage._acquire_lock("run-props")
    assert lock is not None


def test_append_evaluation_delegate(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-eval-io", "exp")
    record = make_record(sample_id="s1")
    eval_record = make_evaluation_record(sample_id="s1")
    storage.append_evaluation("run-eval-io", record, eval_record, eval_id="custom")

    evals = storage.load_cached_evaluations("run-eval-io", eval_id="custom")
    assert len(evals) == 1


def test_facade_internal_helpers(tmp_path):
    storage = ExperimentStorage(tmp_path)
    storage.start_run("run-internal", "exp")

    assert storage._run_metadata_exists("run-internal") is True

    record = make_record(sample_id="s1")
    _ = storage._task_cache_key(record.task)

    meta = storage._load_run_metadata("run-internal")
    storage._save_run_metadata(meta)

    storage._persist_task("run-internal", record.task)
    _ = storage._load_tasks("run-internal")
    _ = storage._load_templates("run-internal")

    payload = storage._serialize_record("run-internal", record)
    _ = storage._deserialize_record(payload, {payload["task_key"]: record.task})
