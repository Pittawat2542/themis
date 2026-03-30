from __future__ import annotations

from pathlib import Path

from themis import Experiment
from themis.core.results import RunStatus
from themis.core.submission import run_batch_request, run_worker_once, submit_experiment


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


def test_submit_experiment_persists_snapshot_and_writes_worker_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root)
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(experiment, config_path=str(config_path), mode="worker_pool")

    assert manifest.run_id == experiment.compile().run_id
    assert manifest.manifest_path == queue_root / "queued" / f"{manifest.run_id}.json"
    assert manifest.manifest_path.is_file()
    assert manifest.status == "pending"

    result = run_worker_once(queue_root)

    assert result is not None
    assert result.status is RunStatus.COMPLETED
    assert (queue_root / "done" / f"{manifest.run_id}.json").is_file()


def test_submit_experiment_writes_batch_request_and_runs_it_later(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root)
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(experiment, config_path=str(config_path), mode="batch")

    assert manifest.manifest_path == batch_root / "requests" / f"{manifest.run_id}.json"
    assert manifest.manifest_path.is_file()

    result = run_batch_request(manifest.manifest_path)

    assert result.status is RunStatus.COMPLETED
    assert (batch_root / "completed" / f"{manifest.run_id}.json").is_file()
