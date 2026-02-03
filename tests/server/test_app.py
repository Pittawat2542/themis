from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from themis.core import entities as core_entities
from themis.experiment import storage as experiment_storage
from themis.server.app import create_app


def _prepare_run(storage: experiment_storage.ExperimentStorage, run_id: str) -> None:
    storage.start_run(run_id, experiment_id="default")

    prompt_spec = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=prompt_spec, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    task = core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": "s1"},
    )
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="ok"),
        error=None,
    )
    eval_record = core_entities.EvaluationRecord(
        sample_id="s1",
        scores=[
            core_entities.MetricScore(metric_name="ExactMatch", value=1.0),
        ],
    )

    storage.append_record(run_id, record, cache_key=experiment_storage.task_cache_key(task))
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
