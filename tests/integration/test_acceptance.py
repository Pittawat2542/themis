from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.subprocess]


def _run_cli(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "themis.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def _data(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1"
    return payload["data"]


def _write_config(
    path: Path, *, store_path: Path, queue_root: Path, batch_root: Path, seed: int = 7
) -> None:
    module = f"{path.stem}_definition"
    path.with_name(f"{module}.py").write_text(
        f"""from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(dataset_id="dataset-1", cases=[Case(
        case_id="case-1", input={{"question": "2+2"}},
        expected_output={{"answer": "4"}},
    )])],
    generation=Generation(generator="builtin/demo_generator", reducer="builtin/majority_vote"),
    evaluation=Evaluation(metrics=["builtin/exact_match"], parser="builtin/json_identity"),
    seeds=[{seed}],
)
"""
    )
    path.write_text(
        f"""
definition: {module}:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
runtime:
  queue_root: {queue_root}
  batch_root: {batch_root}
""".strip()
    )


def test_acceptance_covers_huggingface_and_run_report_export(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )

    fake_datasets_root = tmp_path / "fakepkgs" / "datasets"
    fake_datasets_root.mkdir(parents=True)
    (fake_datasets_root / "__init__.py").write_text(
        """
def load_dataset(dataset_name, *, split):
    return [{"id": "row-1", "prompt": {"question": "2+2"}, "answer": {"answer": "4"}}]
""".strip()
    )
    huggingface = _run_cli(
        "quick-eval",
        "huggingface",
        "--dataset",
        "demo",
        "--split",
        "train",
        "--input-field",
        "prompt",
        "--expected-output-field",
        "answer",
        "--case-id-field",
        "id",
        env={
            "PYTHONPATH": f"{tmp_path / 'fakepkgs'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        },
    )
    run = _run_cli("run", "--config", str(config_path))
    report = _run_cli("report", "--config", str(config_path), "--format", "json")
    export_generation = _run_cli("export", "generation", "--config", str(config_path))

    for result in (huggingface, run, report, export_generation):
        assert result.returncode == 0, result.stderr

    assert _data(huggingface)["status"] == "completed"
    assert _data(run)["status"] == "completed"
    report_data = _data(report)
    assert isinstance(report_data["run_result"], dict)
    assert report_data["run_result"]["status"] == "completed"
    assert report_data["run_result"]["run_id"] == _data(run)["run_id"]
    assert json.loads(export_generation.stdout)["run_id"] == _data(run)["run_id"]


def test_acceptance_covers_worker_pool_and_batch_execution(tmp_path: Path) -> None:
    worker_config = tmp_path / "worker.yaml"
    batch_config = tmp_path / "batch.yaml"
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        worker_config,
        store_path=store_path,
        queue_root=queue_root,
        batch_root=batch_root,
        seed=7,
    )
    _write_config(
        batch_config,
        store_path=store_path,
        queue_root=queue_root,
        batch_root=batch_root,
        seed=8,
    )

    worker_submit = _run_cli(
        "submit", "--config", str(worker_config), "--mode", "worker-pool"
    )
    assert worker_submit.returncode == 0, worker_submit.stderr
    worker_run = _run_cli(
        "worker",
        "run",
        "--queue-root",
        str(queue_root),
        "--definition-root",
        str(tmp_path),
    )
    assert worker_run.returncode == 0, worker_run.stderr
    assert _data(worker_run)["status"] == "completed"

    batch_submit = _run_cli("submit", "--config", str(batch_config), "--mode", "batch")
    assert batch_submit.returncode == 0, batch_submit.stderr
    batch_run = _run_cli(
        "batch",
        "run",
        "--request",
        str(_data(batch_submit)["manifest_path"]),
        "--definition-root",
        str(tmp_path),
    )
    assert batch_run.returncode == 0, batch_run.stderr
    assert _data(batch_run)["status"] == "completed"
