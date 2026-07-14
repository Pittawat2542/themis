from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli.helpers import run_cli
from themis.core.registry import RunLineage
from themis.core.stores.factory import create_run_store
from themis.launcher import _load_runtime_experiment


pytestmark = pytest.mark.slow


def _cli_data(output: str):
    envelope = json.loads(output)
    assert envelope["schema_version"] == "1"
    return envelope["data"]


def _write_config(path: Path, *, store_path: Path) -> None:
    path.with_name("definition.py").write_text(
        """from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(dataset_id="dataset-1", revision="r1", cases=[Case(
        case_id="case-1", input={"question": "2+2"},
        expected_output={"answer": "4"},
    )])],
    generation=Generation(generator="builtin/demo_generator", reducer="builtin/majority_vote"),
    evaluation=Evaluation(metrics=["builtin/exact_match"], parser="builtin/json_identity"),
    seeds=[7],
)
"""
    )
    path.write_text(
        f"""
definition: definition:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
""".strip()
    )


def _write_judge_config(path: Path, *, store_path: Path) -> None:
    path.with_name("judge_definition.py").write_text(
        """from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(dataset_id="dataset-1", revision="r1", cases=[Case(
        case_id="case-1", input={"question": "2+2"},
        expected_output={"answer": "4"},
    )])],
    generation=Generation(generator="builtin/demo_generator", reducer="builtin/majority_vote"),
    evaluation=Evaluation(
        metrics=["builtin/llm_rubric"], parser="builtin/json_identity",
        judge_models=["builtin/demo_judge"],
        workflow_options={"rubric": "pass if the answer is correct"},
    ),
    seeds=[7],
)
"""
    )
    path.write_text(
        f"""
definition: judge_definition:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
""".strip()
    )


def test_run_resume_estimate_and_quickcheck_use_config_driven_experiments(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    estimate = run_cli("estimate", "--config", str(config_path))
    assert estimate.returncode == 0, estimate.stderr
    estimate_payload = _cli_data(estimate.stdout)
    assert estimate_payload["planned_generation_tasks"] == 1

    run = run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr
    run_payload = _cli_data(run.stdout)
    assert run_payload["status"] == "completed"

    resume = run_cli("resume", "--config", str(config_path))
    assert resume.returncode == 0, resume.stderr
    resume_payload = _cli_data(resume.stdout)
    assert resume_payload["run_id"] == run_payload["run_id"]
    assert resume_payload["status"] == "completed"

    quickcheck = run_cli("quickcheck", "--config", str(config_path))
    assert quickcheck.returncode == 0, quickcheck.stderr
    quickcheck_payload = _cli_data(quickcheck.stdout)
    assert quickcheck_payload["run_id"] == run_payload["run_id"]
    assert quickcheck_payload["metric_means"] == {"builtin/exact_match": 1.0}


def test_run_supports_stage_limited_execution(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    run = run_cli("run", "--config", str(config_path), "--until-stage", "generate")
    assert run.returncode == 0, run.stderr
    run_payload = _cli_data(run.stdout)

    assert run_payload["status"] == "completed"
    assert run_payload["completed_through_stage"] == "generate"

    resume = run_cli("resume", "--config", str(config_path))
    assert resume.returncode == 0, resume.stderr
    resume_payload = _cli_data(resume.stdout)
    assert resume_payload["completed_through_stage"] == "generate"


def test_inspect_and_replay_commands_expose_persisted_state(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_config(config_path, store_path=store_path)

    run = run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr
    run_payload = _cli_data(run.stdout)
    run_id = run_payload["run_id"]

    store = create_run_store(_load_runtime_experiment(config_path).storage)
    store.initialize()
    store.update_run_record(run_id, tags=["phase1", "smoke"], baseline_label="main")

    inspect_snapshot = run_cli("inspect", "snapshot", "--config", str(config_path))
    assert inspect_snapshot.returncode == 0, inspect_snapshot.stderr
    snapshot_payload = _cli_data(inspect_snapshot.stdout)
    assert snapshot_payload["run_id"] == run_id

    inspect_state = run_cli("inspect", "state", "--config", str(config_path))
    assert inspect_state.returncode == 0, inspect_state.stderr
    state_payload = _cli_data(inspect_state.stdout)
    assert state_payload["run_id"] == run_id
    assert state_payload["status"] == "completed"

    inspect_runs = run_cli(
        "inspect", "runs", "--config", str(config_path), "--tag", "phase1"
    )
    assert inspect_runs.returncode == 0, inspect_runs.stderr
    assert _cli_data(inspect_runs.stdout)[0]["run_id"] == run_id

    inspect_record = run_cli(
        "inspect",
        "run-record",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    )
    assert inspect_record.returncode == 0, inspect_record.stderr
    assert _cli_data(inspect_record.stdout)["baseline_label"] == "main"

    store.update_run_record(
        run_id,
        lineage=[RunLineage(parent_run_id="parent-run", relationship="rerun")],
    )
    inspect_lineage = run_cli(
        "inspect", "lineage", "--config", str(config_path), "--run-id", run_id
    )
    assert inspect_lineage.returncode == 0, inspect_lineage.stderr
    lineage_payload = _cli_data(inspect_lineage.stdout)
    assert lineage_payload["run_id"] == run_id
    assert lineage_payload["lineage"][0]["parent_run_id"] == "parent-run"

    inspect_case = run_cli(
        "inspect",
        "case",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--case-id",
        "case-1",
        "--dataset-id",
        "dataset-1",
    )
    assert inspect_case.returncode == 0, inspect_case.stderr
    assert _cli_data(inspect_case.stdout)["case_id"] == "case-1"

    inspect_telemetry = run_cli(
        "inspect", "telemetry", "--config", str(config_path), "--run-id", run_id
    )
    assert inspect_telemetry.returncode == 0, inspect_telemetry.stderr
    assert _cli_data(inspect_telemetry.stdout)["run_id"] == run_id

    missing_evaluation = run_cli(
        "inspect",
        "evaluation",
        "--config",
        str(config_path),
        "--case-id",
        "case-1",
        "--metric-id",
        "builtin/llm_rubric",
        "--dataset-id",
        "dataset-1",
    )
    assert missing_evaluation.returncode != 0
    assert "No evaluation execution found" in missing_evaluation.stderr
    assert "dataset_id=dataset-1" in missing_evaluation.stderr

    replay = run_cli("replay", "--config", str(config_path), "--stage", "score")
    assert replay.returncode == 0, replay.stderr
    replay_payload = _cli_data(replay.stdout)
    assert replay_payload["run_id"] == run_id
    assert replay_payload["status"] == "completed"

    rerun = run_cli(
        "rerun",
        "--config",
        str(config_path),
        "--stage",
        "score",
        "--case-id",
        "case-1",
        "--metric-id",
        "builtin/exact_match",
    )
    assert rerun.returncode == 0, rerun.stderr
    rerun_payload = _cli_data(rerun.stdout)
    assert rerun_payload["run_id"] == run_id
    assert rerun_payload["status"] == "completed"


def test_inspect_evaluation_returns_workflow_execution(tmp_path: Path) -> None:
    config_path = tmp_path / "judge-experiment.yaml"
    store_path = tmp_path / "run.sqlite3"
    _write_judge_config(config_path, store_path=store_path)

    run = run_cli("run", "--config", str(config_path))
    assert run.returncode == 0, run.stderr

    inspect_evaluation = run_cli(
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
    payload = _cli_data(inspect_evaluation.stdout)
    assert payload["execution_id"]
    assert payload["metric_results"][0]["metric_id"] == "builtin/llm_rubric"
