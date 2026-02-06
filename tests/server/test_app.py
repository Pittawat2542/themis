from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from themis.experiment import storage as experiment_storage
from themis.server.app import create_app
from tests.factories import make_evaluation_record, make_record


def _prepare_run(storage: experiment_storage.ExperimentStorage, run_id: str) -> None:
    storage.start_run(run_id, experiment_id="default")

    record = make_record(sample_id="s1")
    eval_record = make_evaluation_record(sample_id="s1", metric_name="ExactMatch", value=1.0)

    storage.append_record(
        run_id,
        record,
        cache_key=experiment_storage.task_cache_key(record.task),
    )
    storage.append_evaluation(run_id, record, eval_record)


def test_server_run_detail_uses_core_entities(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)
    run_id = "run-1"
    _prepare_run(storage, run_id)

    app = create_app(storage_path=tmp_path)
    client = TestClient(app)

    resp = client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["run_id"] == run_id
    assert payload["num_samples"] == 1
    assert payload["metrics"]["ExactMatch"] == 1.0
    assert payload["samples"][0]["id"] == "s1"
    assert payload["samples"][0]["prompt"] == "Q"
    assert payload["samples"][0]["response"] == "ok"


def test_server_runs_list(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)
    _prepare_run(storage, "run-2")

    app = create_app(storage_path=tmp_path)
    client = TestClient(app)

    resp = client.get("/api/runs")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload
    assert payload[0]["metrics"]["ExactMatch"] == 1.0
    assert payload[0]["status"] == "in_progress"
    assert payload[0]["experiment_id"] == "default"


def test_server_compare_endpoint(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)
    _prepare_run(storage, "run-a")

    storage.start_run("run-b", experiment_id="default")
    record = make_record(sample_id="s1")
    eval_record = make_evaluation_record(
        sample_id="s1", metric_name="ExactMatch", value=0.0
    )
    storage.append_record(
        "run-b",
        record,
        cache_key=experiment_storage.task_cache_key(record.task),
    )
    storage.append_evaluation("run-b", record, eval_record)

    app = create_app(storage_path=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/compare",
        json={
            "run_ids": ["run-a", "run-b"],
            "metrics": ["ExactMatch"],
            "statistical_test": "bootstrap",
            "alpha": 0.05,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["metrics"] == ["ExactMatch"]
    assert len(payload["pairwise_results"]) == 1


def test_server_compare_endpoint_rejects_invalid_input(tmp_path):
    storage = experiment_storage.ExperimentStorage(tmp_path)
    _prepare_run(storage, "run-a")

    app = create_app(storage_path=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/compare",
        json={
            "run_ids": ["run-a"],
            "metrics": ["ExactMatch"],
        },
    )
    assert resp.status_code == 400
