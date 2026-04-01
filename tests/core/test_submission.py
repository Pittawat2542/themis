from __future__ import annotations

import os
from pathlib import Path

import pytest

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, GenerationResult
from themis.core.results import RunStatus
from themis.core.submission import run_batch_request, run_worker_once, submit_experiment


def _write_config(
    path: Path, *, store_path: Path, queue_root: Path, batch_root: Path
) -> None:
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


class SubmissionConfigGenerator:
    component_id = "generator/submission-config"
    version = "1.0"

    def fingerprint(self) -> str:
        return "submission-config-generator"

    async def generate(self, case: Case, ctx: object) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate", final_output=case.expected_output
        )


SUBMISSION_CONFIG_GENERATOR = SubmissionConfigGenerator()


def test_submit_experiment_persists_snapshot_and_writes_worker_manifest(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(
        experiment, config_path=str(config_path), mode="worker_pool"
    )

    assert manifest.run_id == experiment.compile().run_id
    assert manifest.manifest_path == queue_root / "queued" / f"{manifest.run_id}.json"
    assert manifest.manifest_path.is_file()
    assert manifest.status == "pending"

    result = run_worker_once(queue_root)

    assert result is not None
    assert result.status is RunStatus.COMPLETED
    assert (queue_root / "done" / f"{manifest.run_id}.json").is_file()


def test_submit_experiment_writes_batch_request_and_runs_it_later(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(experiment, config_path=str(config_path), mode="batch")

    assert manifest.manifest_path == batch_root / "requests" / f"{manifest.run_id}.json"
    assert manifest.manifest_path.is_file()

    result = run_batch_request(manifest.manifest_path)

    assert result.status is RunStatus.COMPLETED
    assert (batch_root / "completed" / f"{manifest.run_id}.json").is_file()


def test_submit_experiment_is_immutable_after_config_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(
        config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root
    )
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(
        experiment, config_path=str(config_path), mode="worker_pool"
    )
    config_path.write_text(config_path.read_text().replace("seeds: [7]", "seeds: [8]"))

    result = run_worker_once(queue_root)

    assert result is not None
    assert result.run_id == manifest.run_id
    assert result.status is RunStatus.COMPLETED


def test_run_worker_once_uses_manifest_when_current_directory_changes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = workspace / "experiment.yaml"
    store_path = workspace / "runs" / "run.sqlite3"
    queue_root = workspace / "queue"
    _write_config(
        config_path,
        store_path=Path("runs/run.sqlite3"),
        queue_root=Path("queue"),
        batch_root=Path("batch"),
    )

    experiment = Experiment.from_config(config_path)
    manifest = submit_experiment(
        experiment, config_path=str(Path("experiment.yaml")), mode="worker_pool"
    )

    original_cwd = Path.cwd()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    os.chdir(other_cwd)
    try:
        result = run_worker_once(queue_root)
    finally:
        os.chdir(original_cwd)

    assert result is not None
    assert result.run_id == manifest.run_id
    assert result.status is RunStatus.COMPLETED
    assert store_path.is_file()


def test_submit_experiment_accepts_config_loaded_importable_components(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
generation:
  generator: tests.core.test_submission:SUBMISSION_CONFIG_GENERATOR
evaluation:
  metrics: []
  parsers: []
storage:
  store: sqlite
  parameters:
    path: {tmp_path / "run.sqlite3"}
runtime:
  queue_root: {tmp_path / "queue"}
  batch_root: {tmp_path / "batch"}
datasets:
  - dataset_id: dataset-1
    cases:
      - case_id: case-1
        input: "hello"
        expected_output: "hello"
""".strip()
    )
    experiment = Experiment.from_config(config_path)

    manifest = submit_experiment(
        experiment, config_path=str(config_path), mode="worker_pool"
    )
    result = run_worker_once(tmp_path / "queue")

    assert manifest.snapshot.run_id == manifest.run_id
    assert result is not None
    assert result.run_id == manifest.run_id
    assert result.status is RunStatus.COMPLETED


def test_submit_experiment_rejects_non_importable_runtime_components(
    tmp_path: Path,
) -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator=SubmissionConfigGenerator(),
            candidate_policy={"num_samples": 1},
        ),
        evaluation=EvaluationConfig(metrics=[], parsers=[]),
        storage=StorageConfig(store="memory"),
        datasets=[],
    )

    with pytest.raises(ValueError, match="importable config symbols"):
        submit_experiment(
            experiment,
            config_path=str(tmp_path / "experiment.yaml"),
            mode="worker_pool",
        )
