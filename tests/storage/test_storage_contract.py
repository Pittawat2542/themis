from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from themis.backends.storage import LocalFileStorageBackend
from themis.storage import ExperimentStorage
from tests.factories import make_evaluation_record, make_record


@dataclass
class _StorageAdapter:
    start_run: callable
    append_generation_record: callable
    append_evaluation_record: callable
    complete_run: callable
    fail_run: callable
    run_exists: callable
    load_status: callable


def _experiment_storage_adapter(root: Path) -> _StorageAdapter:
    storage = ExperimentStorage(root)

    return _StorageAdapter(
        start_run=lambda run_id: storage.start_run(run_id, "exp-contract", config={}),
        append_generation_record=lambda run_id, record: storage.append_record(
            run_id, record
        ),
        append_evaluation_record=lambda run_id,
        record,
        evaluation: storage.append_evaluation(run_id, record, evaluation),
        complete_run=lambda run_id: storage.complete_run(run_id),
        fail_run=lambda run_id, error: storage.fail_run(run_id, error),
        run_exists=lambda run_id: storage.run_metadata_exists(run_id),
        load_status=lambda run_id: storage._load_run_metadata(run_id).status.value,
    )


def _local_backend_adapter(root: Path) -> _StorageAdapter:
    backend = LocalFileStorageBackend(root)

    return _StorageAdapter(
        start_run=lambda run_id: backend.start_run(
            run_id, experiment_id="exp-contract", config={}
        ),
        append_generation_record=lambda run_id,
        record: backend.append_generation_record(run_id, record),
        append_evaluation_record=lambda run_id,
        record,
        evaluation: backend.append_evaluation_record(run_id, record, evaluation),
        complete_run=lambda run_id: backend.complete_run(run_id),
        fail_run=lambda run_id, error: backend.fail_run(run_id, error),
        run_exists=lambda run_id: backend.run_exists(run_id),
        load_status=lambda run_id: backend.load_run_metadata(run_id)["status"],
    )


@pytest.mark.parametrize(
    "adapter_factory",
    [_experiment_storage_adapter, _local_backend_adapter],
)
def test_storage_contract_lifecycle_complete(tmp_path, adapter_factory):
    adapter = adapter_factory(tmp_path / adapter_factory.__name__)
    run_id = "run-complete"
    record = make_record(sample_id="s1")
    evaluation = make_evaluation_record(sample_id="s1")

    adapter.start_run(run_id)
    adapter.append_generation_record(run_id, record)
    adapter.append_evaluation_record(run_id, record, evaluation)
    adapter.complete_run(run_id)

    assert adapter.run_exists(run_id) is True
    assert adapter.load_status(run_id) == "completed"


@pytest.mark.parametrize(
    "adapter_factory",
    [_experiment_storage_adapter, _local_backend_adapter],
)
def test_storage_contract_lifecycle_fail(tmp_path, adapter_factory):
    adapter = adapter_factory(tmp_path / adapter_factory.__name__)
    run_id = "run-failed"

    adapter.start_run(run_id)
    adapter.fail_run(run_id, "boom")

    assert adapter.run_exists(run_id) is True
    assert adapter.load_status(run_id) == "failed"
