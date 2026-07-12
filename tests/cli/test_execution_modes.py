from __future__ import annotations

import json
from pathlib import Path

from tests.cli.helpers import run_cli


def _write_config(
    path: Path, *, store_path: Path, queue_root: Path, batch_root: Path
) -> None:
    definition_path = path.with_name("experiment_definition.py")
    definition_path.write_text(
        """from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="dataset-1",
        cases=[Case(
            case_id="case-1",
            input={"question": "2+2"},
            expected_output={"answer": "4"},
        )],
    )],
    generation=Generation(
        generator="builtin/demo_generator",
        reducer="builtin/majority_vote",
    ),
    evaluation=Evaluation(
        metrics=["builtin/exact_match"],
        parser="builtin/json_identity",
    ),
    seeds=[7],
)
"""
    )
    path.write_text(
        f"""
definition: experiment_definition:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
runtime:
  queue_root: {queue_root}
  batch_root: {batch_root}
""".strip()
    )


def test_worker_pool_submit_resume_and_run(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )

    submit = run_cli("submit", "--config", str(config_path), "--mode", "worker-pool")
    assert submit.returncode == 0, submit.stderr
    submit_payload = json.loads(submit.stdout)
    assert submit_payload["status"] == "pending"

    resume_pending = run_cli("resume", "--config", str(config_path))
    assert resume_pending.returncode == 0, resume_pending.stderr
    assert json.loads(resume_pending.stdout)["status"] == "pending"

    worker_run = run_cli(
        "worker",
        "run",
        "--queue-root",
        str(queue_root),
        "--definition-root",
        str(tmp_path),
    )
    assert worker_run.returncode == 0, worker_run.stderr
    assert json.loads(worker_run.stdout)["status"] == "completed"


def test_batch_submit_resume_and_run_request(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )

    submit = run_cli("submit", "--config", str(config_path), "--mode", "batch")
    assert submit.returncode == 0, submit.stderr
    submit_payload = json.loads(submit.stdout)
    assert submit_payload["status"] == "pending"

    resume_pending = run_cli("resume", "--config", str(config_path))
    assert resume_pending.returncode == 0, resume_pending.stderr
    assert json.loads(resume_pending.stdout)["status"] == "pending"

    batch_run = run_cli(
        "batch",
        "run",
        "--request",
        submit_payload["manifest_path"],
        "--definition-root",
        str(tmp_path),
    )
    assert batch_run.returncode == 0, batch_run.stderr
    assert json.loads(batch_run.stdout)["status"] == "completed"


def test_submit_rejects_invalid_mode_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )

    result = run_cli("submit", "--config", str(config_path), "--mode", "typo")

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "worker-pool" in result.stderr
    assert "batch" in result.stderr
