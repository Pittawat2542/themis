from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from themis import storage as experiment_storage
from themis.server.app import create_app
from tests.factories import make_evaluation_record, make_record


@pytest.fixture
def tmp_storage(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)

    # Run 1: Completed run
    run1 = "run-1"
    storage.start_run(run1, experiment_id="default")
    record1 = make_record(sample_id="s1")
    eval_record1 = make_evaluation_record(
        sample_id="s1", metric_name="ExactMatch", value=1.0
    )
    storage.append_record(
        run1, record1, cache_key=experiment_storage.task_cache_key(record1.task)
    )
    storage.append_evaluation(run1, record1, eval_record1)
    # Important: finalize run 1 so status is "completed"
    storage.complete_run(run1)

    # Run 2: In-progress run
    run2 = "run-2"
    storage.start_run(run2, experiment_id="default")
    record2 = make_record(sample_id="s2")
    storage.append_record(
        run2, record2, cache_key=experiment_storage.task_cache_key(record2.task)
    )

    return tmp_path, storage


@pytest.fixture
def test_app(tmp_storage):
    tmp_path, _ = tmp_storage
    return create_app(storage_path=tmp_path)


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
