from __future__ import annotations

from pathlib import Path

import pytest

from themis.core.submission import (
    run_batch_request,
    run_worker_once,
    submit_experiment,
)
from themis.launcher import _load_runtime_experiment


def _write_launcher(root: Path, *, seed: int = 7) -> Path:
    (root / "definition.py").write_text(
        f"""from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="dataset-1",
        cases=[Case(
            case_id="case-1",
            input={{"answer": "4"}},
            expected_output={{"answer": "4"}},
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
    seeds=[{seed}],
)
""",
        encoding="utf-8",
    )
    launcher = root / "experiment.yaml"
    launcher.write_text(
        """definition: definition:experiment
storage:
  target: sqlite
  kwargs:
    path: runs.sqlite3
runtime:
  queue_root: queue
  batch_root: batch
""",
        encoding="utf-8",
    )
    return launcher


def test_worker_submission_round_trips_python_definition(tmp_path: Path) -> None:
    launcher = _write_launcher(tmp_path)
    experiment = _load_runtime_experiment(launcher)

    manifest = submit_experiment(
        experiment, config_path=str(launcher), mode="worker_pool"
    )
    result = run_worker_once(tmp_path / "queue", definition_roots=[tmp_path])

    assert result is not None
    assert result.run_id == manifest.run_id
    assert result.status.value == "completed"
    assert (tmp_path / "queue" / "done" / f"{manifest.run_id}.json").is_file()


def test_batch_submission_round_trips_python_definition(tmp_path: Path) -> None:
    launcher = _write_launcher(tmp_path)
    experiment = _load_runtime_experiment(launcher)
    manifest = submit_experiment(experiment, config_path=str(launcher), mode="batch")

    result = run_batch_request(manifest.manifest_path, definition_roots=[tmp_path])

    assert result.run_id == manifest.run_id
    assert result.status.value == "completed"


def test_worker_rejects_definition_drift_after_submission(tmp_path: Path) -> None:
    launcher = _write_launcher(tmp_path)
    experiment = _load_runtime_experiment(launcher)
    submit_experiment(experiment, config_path=str(launcher), mode="worker_pool")
    _write_launcher(tmp_path, seed=8)

    with pytest.raises(ValueError, match="digest no longer matches"):
        run_worker_once(tmp_path / "queue", definition_roots=[tmp_path])


def test_manifest_uses_absolute_launcher_path(tmp_path: Path, monkeypatch) -> None:
    launcher = _write_launcher(tmp_path)
    experiment = _load_runtime_experiment(launcher)
    manifest = submit_experiment(
        experiment, config_path=str(launcher), mode="worker_pool"
    )
    monkeypatch.chdir(tmp_path.parent)

    result = run_worker_once(tmp_path / "queue", definition_roots=[tmp_path])

    assert Path(manifest.config_path).is_absolute()
    assert result is not None


def test_submission_requires_accessible_launcher(tmp_path: Path) -> None:
    launcher = _write_launcher(tmp_path)
    experiment = _load_runtime_experiment(launcher)

    with pytest.raises(ValueError, match="accessible launcher"):
        submit_experiment(
            experiment,
            config_path=str(tmp_path / "missing.yaml"),
            mode="worker_pool",
        )
