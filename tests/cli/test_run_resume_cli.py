from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_config(path: Path, *, store_path: Path) -> None:
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
datasets:
  - dataset_id: dataset-1
    revision: r1
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
seeds: [7]
""".strip()
    )


def _write_judge_config(path: Path, *, store_path: Path) -> None:
    path.write_text(
        f"""
generation:
  generator: builtin/demo_generator
  candidate_policy:
    num_samples: 1
  reducer: builtin/majority_vote
evaluation:
  metrics:
    - builtin/llm_rubric
  parsers:
    - builtin/json_identity
  judge_models:
    - builtin/demo_judge
  workflow_overrides:
    rubric: pass if the answer is correct
storage:
  store: sqlite
  parameters:
    path: {store_path}
datasets:
  - dataset_id: dataset-1
    revision: r1
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


def test_run_resume_estimate_and_quickcheck_use_config_driven_experiments(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    estimate = _run_cli("estimate", "--config", str(config_path))
    assert estimate.returncode == 0, estimate.stderr
    estimate_payload = json.loads(estimate.stdout)
    assert estimate_payload["planned_generation_tasks"] == 1

    run = _run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr
    run_payload = json.loads(run.stdout)
    assert run_payload["status"] == "completed"

    resume = _run_cli("resume", "--config", str(config_path))
    assert resume.returncode == 0, resume.stderr
    resume_payload = json.loads(resume.stdout)
    assert resume_payload["run_id"] == run_payload["run_id"]
    assert resume_payload["status"] == "completed"

    quickcheck = _run_cli("quickcheck", "--config", str(config_path))
    assert quickcheck.returncode == 0, quickcheck.stderr
    quickcheck_payload = json.loads(quickcheck.stdout)
    assert quickcheck_payload["run_id"] == run_payload["run_id"]
    assert quickcheck_payload["metric_means"] == {"builtin/exact_match": 1.0}


def test_run_supports_stage_limited_execution(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    run = _run_cli("run", "--config", str(config_path), "--until-stage", "generate")
    assert run.returncode == 0, run.stderr
    run_payload = json.loads(run.stdout)

    assert run_payload["status"] == "completed"
    assert run_payload["completed_through_stage"] == "generate"

    resume = _run_cli("resume", "--config", str(config_path))
    assert resume.returncode == 0, resume.stderr
    resume_payload = json.loads(resume.stdout)
    assert resume_payload["completed_through_stage"] == "generate"


def test_inspect_and_replay_commands_expose_persisted_state(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    run = _run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr
    run_payload = json.loads(run.stdout)

    inspect_snapshot = _run_cli("inspect", "snapshot", "--config", str(config_path))
    assert inspect_snapshot.returncode == 0, inspect_snapshot.stderr
    snapshot_payload = json.loads(inspect_snapshot.stdout)
    assert snapshot_payload["run_id"] == run_payload["run_id"]

    inspect_state = _run_cli("inspect", "state", "--config", str(config_path))
    assert inspect_state.returncode == 0, inspect_state.stderr
    state_payload = json.loads(inspect_state.stdout)
    assert state_payload["run_id"] == run_payload["run_id"]
    assert state_payload["status"] == "completed"

    replay = _run_cli("replay", "--config", str(config_path), "--stage", "score")
    assert replay.returncode == 0, replay.stderr
    replay_payload = json.loads(replay.stdout)
    assert replay_payload["run_id"] == run_payload["run_id"]
    assert replay_payload["status"] == "completed"


def test_inspect_evaluation_returns_workflow_execution(tmp_path: Path) -> None:
    config_path = tmp_path / "judge-experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_judge_config(config_path, store_path=store_path)

    run = _run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr

    inspect_evaluation = _run_cli(
        "inspect",
        "evaluation",
        "--config",
        str(config_path),
        "--case-id",
        "case-1",
        "--metric-id",
        "builtin/llm_rubric",
    )
    assert inspect_evaluation.returncode == 0, inspect_evaluation.stderr
    payload = json.loads(inspect_evaluation.stdout)
    assert payload["execution_id"]
    assert payload["scores"][0]["metric_id"] == "builtin/llm_rubric"
