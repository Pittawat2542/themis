from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_config(path: Path, *, store_path: Path, answer: str, seed: int) -> None:
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
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "{answer}"
seeds: [{seed}]
""".strip()
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "themis.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_report_compare_and_export_commands_use_existing_read_side_helpers(
    tmp_path: Path,
) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    store_path = tmp_path / "runs.sqlite3"
    _write_config(baseline_config, store_path=store_path, answer="4", seed=7)
    _write_config(candidate_config, store_path=store_path, answer="4", seed=8)

    baseline_run = _run_cli("run", "--config", str(baseline_config))
    candidate_run = _run_cli("run", "--config", str(candidate_config))
    assert baseline_run.returncode == 0, baseline_run.stderr
    assert candidate_run.returncode == 0, candidate_run.stderr

    report_json = _run_cli(
        "report", "--config", str(baseline_config), "--format", "json"
    )
    assert report_json.returncode == 0, report_json.stderr
    report_payload = json.loads(report_json.stdout)
    assert report_payload["run_result"]["status"] == "completed"
    assert (
        report_payload["snapshot"]["run_id"]
        == json.loads(baseline_run.stdout)["run_id"]
    )
    assert (
        report_payload["execution_state"]["run_id"]
        == json.loads(baseline_run.stdout)["run_id"]
    )

    report_markdown = _run_cli(
        "report", "--config", str(baseline_config), "--format", "markdown"
    )
    assert report_markdown.returncode == 0, report_markdown.stderr
    assert "# Run Report" in report_markdown.stdout

    report_csv = _run_cli("report", "--config", str(baseline_config), "--format", "csv")
    assert report_csv.returncode == 0, report_csv.stderr
    assert (
        report_csv.stdout.splitlines()[0]
        == "case_id,dataset_id,case_key,metric_id,outcome,value,candidate_id,error_category,error_message,details"
    )

    report_latex = _run_cli(
        "report", "--config", str(baseline_config), "--format", "latex"
    )
    assert report_latex.returncode == 0, report_latex.stderr
    assert "\\begin{tabular}" in report_latex.stdout

    compare = _run_cli(
        "compare",
        "--baseline-config",
        str(baseline_config),
        "--candidate-config",
        str(candidate_config),
    )
    assert compare.returncode == 0, compare.stderr
    compare_payload = json.loads(compare.stdout)
    assert compare_payload["metrics"]["builtin/exact_match"]["ties"] == 1

    generation_export = _run_cli(
        "export", "generation", "--config", str(baseline_config)
    )
    assert generation_export.returncode == 0, generation_export.stderr
    generation_payload = json.loads(generation_export.stdout)
    assert generation_payload["run_id"] == json.loads(baseline_run.stdout)["run_id"]

    evaluation_export = _run_cli(
        "export", "evaluation", "--config", str(baseline_config)
    )
    assert evaluation_export.returncode == 0, evaluation_export.stderr
    evaluation_payload = json.loads(evaluation_export.stdout)
    assert evaluation_payload["run_id"] == json.loads(baseline_run.stdout)["run_id"]
