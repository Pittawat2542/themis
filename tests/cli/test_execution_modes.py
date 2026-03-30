from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_config(path: Path, *, store_path: Path, queue_root: Path, batch_root: Path) -> None:
    path.write_text(
        f"""
generation:
  generator: builtin/demo_generator
  candidate_policy:
    num_samples: 1
  reducer: builtin/majority_vote
evaluation:
  metrics:
    - builtin/exact_match
  parsers:
    - builtin/json_identity
storage:
  store: sqlite
  parameters:
    path: {store_path}
runtime:
  queue_root: {queue_root}
  batch_root: {batch_root}
datasets:
  - dataset_id: dataset-1
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
seeds: [7]
""".strip()
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "themis.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_worker_pool_submit_resume_and_run(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root)

    submit = _run_cli("submit", "--config", str(config_path), "--mode", "worker-pool")
    assert submit.returncode == 0, submit.stderr
    submit_payload = json.loads(submit.stdout)
    assert submit_payload["status"] == "pending"

    resume_pending = _run_cli("resume", "--config", str(config_path))
    assert resume_pending.returncode == 0, resume_pending.stderr
    assert json.loads(resume_pending.stdout)["status"] == "pending"

    worker_run = _run_cli("worker", "run", "--queue-root", str(queue_root), "--once")
    assert worker_run.returncode == 0, worker_run.stderr
    assert json.loads(worker_run.stdout)["status"] == "completed"


def test_batch_submit_resume_and_run_request(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root)

    submit = _run_cli("submit", "--config", str(config_path), "--mode", "batch")
    assert submit.returncode == 0, submit.stderr
    submit_payload = json.loads(submit.stdout)
    assert submit_payload["status"] == "pending"

    resume_pending = _run_cli("resume", "--config", str(config_path))
    assert resume_pending.returncode == 0, resume_pending.stderr
    assert json.loads(resume_pending.stdout)["status"] == "pending"

    batch_run = _run_cli("batch", "run", "--request", submit_payload["manifest_path"])
    assert batch_run.returncode == 0, batch_run.stderr
    assert json.loads(batch_run.stdout)["status"] == "completed"
