from __future__ import annotations

import pytest


def test_list_runs(client):
    response = client.get("/api/runs")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Check structure
    run1 = next(r for r in data if r["run_id"] == "run-1")
    assert run1["status"] == "completed"
    assert run1["experiment_id"] == "default"
    assert "ExactMatch" in run1["metrics"]
    assert run1["metrics"]["ExactMatch"] == 1.0


def test_get_run_detail(client):
    response = client.get("/api/runs/run-1")
    assert response.status_code == 200

    data = response.json()
    assert data["run_id"] == "run-1"
    assert data["status"] == "completed"
    assert "ExactMatch" in data["metrics"]
    assert len(data["samples"]) == 1
    assert data["samples"][0]["id"] == "s1"


def test_get_nonexistent_run(client):
    response = client.get("/api/runs/nonexistent-run")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.skip(reason="Not implemented yet but expected in TDD")
def test_get_run_records(client):
    """Test generating a paginated or full list of generation records."""
    response = client.get("/api/runs/run-1/records")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["task"]["metadata"]["dataset_id"] == "s1"
