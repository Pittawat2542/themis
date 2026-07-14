from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli.helpers import run_cli
from themis.core.stores.factory import create_run_store
from themis.launcher import _load_runtime_experiment


pytestmark = pytest.mark.slow


def _cli_data(output: str):
    envelope = json.loads(output)
    assert envelope["schema_version"] == "1"
    return envelope["data"]


def _write_config(path: Path, *, store_path: Path, answer: str, seed: int) -> None:
    module = f"{path.stem}_definition"
    path.with_name(f"{module}.py").write_text(
        f'''from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(dataset_id="dataset-1", cases=[Case(
        case_id="case-1", input={{"question": "2+2"}},
        expected_output={{"answer": "{answer}"}},
    )])],
    generation=Generation(generator="builtin/demo_generator", reducer="builtin/majority_vote"),
    evaluation=Evaluation(metrics=["builtin/exact_match"], parser="builtin/json_identity"),
    seeds=[{seed}],
)
'''
    )
    path.write_text(
        f"""
definition: {module}:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
""".strip()
    )


def test_report_compare_and_export_commands_use_existing_read_side_helpers(
    tmp_path: Path,
) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    store_path = tmp_path / "runs.sqlite3"
    _write_config(baseline_config, store_path=store_path, answer="4", seed=7)
    _write_config(candidate_config, store_path=store_path, answer="4", seed=8)

    baseline_run = run_cli("run", "--config", str(baseline_config))
    candidate_run = run_cli("run", "--config", str(candidate_config))
    assert baseline_run.returncode == 0, baseline_run.stderr
    assert candidate_run.returncode == 0, candidate_run.stderr
    baseline_run_id = _cli_data(baseline_run.stdout)["run_id"]
    candidate_run_id = _cli_data(candidate_run.stdout)["run_id"]

    store = create_run_store(_load_runtime_experiment(baseline_config).storage)
    store.initialize()
    store.update_run_record(baseline_run_id, baseline_label="main")
    store.update_run_record(candidate_run_id, baseline_label="candidate")

    report_json = run_cli(
        "report", "--config", str(baseline_config), "--format", "json"
    )
    assert report_json.returncode == 0, report_json.stderr
    report_payload = _cli_data(report_json.stdout)
    assert report_payload["run_result"]["status"] == "completed"
    assert report_payload["snapshot"]["run_id"] == baseline_run_id
    assert report_payload["execution_state"]["run_id"] == baseline_run_id

    report_markdown = run_cli(
        "report", "--config", str(baseline_config), "--format", "markdown"
    )
    assert report_markdown.returncode == 0, report_markdown.stderr
    assert "# Run Report" in report_markdown.stdout

    report_csv = run_cli("report", "--config", str(baseline_config), "--format", "csv")
    assert report_csv.returncode == 0, report_csv.stderr
    assert (
        report_csv.stdout.splitlines()[0]
        == "metric_id,count,mean,min,max,ci_lower,ci_upper"
    )

    report_latex = run_cli(
        "report", "--config", str(baseline_config), "--format", "latex"
    )
    assert report_latex.returncode == 0, report_latex.stderr
    assert "\\begin{tabular}" in report_latex.stdout

    compare = run_cli(
        "compare",
        "--baseline-config",
        str(baseline_config),
        "--candidate-config",
        str(candidate_config),
    )
    assert compare.returncode == 0, compare.stderr
    compare_payload = _cli_data(compare.stdout)
    assert compare_payload["metrics"][0]["metric_id"] == "builtin/exact_match"
    assert compare_payload["metrics"][0]["ties"] == 1

    compare_run_ids = run_cli(
        "compare-runs",
        "--config",
        str(baseline_config),
        "--baseline-run-id",
        baseline_run_id,
        "--candidate-run-id",
        candidate_run_id,
    )
    assert compare_run_ids.returncode == 0, compare_run_ids.stderr
    compare_run_ids_payload = _cli_data(compare_run_ids.stdout)
    assert compare_run_ids_payload["evidence_run_ids"] == [
        baseline_run_id,
        candidate_run_id,
    ]
    assert compare_run_ids_payload["metrics"][0]["pairs"] == 1

    compare_labels = run_cli(
        "compare-latest",
        "--config",
        str(baseline_config),
        "--baseline-label",
        "main",
        "--candidate-label",
        "candidate",
    )
    assert compare_labels.returncode == 0, compare_labels.stderr
    compare_labels_payload = _cli_data(compare_labels.stdout)
    assert compare_labels_payload["evidence_run_ids"] == [
        baseline_run_id,
        candidate_run_id,
    ]
    assert compare_labels_payload["metrics"][0]["pairs"] == 1

    generation_export = run_cli(
        "export", "generation", "--config", str(baseline_config)
    )
    assert generation_export.returncode == 0, generation_export.stderr
    generation_payload = json.loads(generation_export.stdout)
    assert generation_payload["run_id"] == baseline_run_id

    evaluation_export = run_cli(
        "export", "evaluation", "--config", str(baseline_config)
    )
    assert evaluation_export.returncode == 0, evaluation_export.stderr
    evaluation_payload = json.loads(evaluation_export.stdout)
    assert evaluation_payload["run_id"] == baseline_run_id
