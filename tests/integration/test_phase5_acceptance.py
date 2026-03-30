from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def _write_config(path: Path, *, store_path: Path, queue_root: Path, batch_root: Path, seed: int = 7) -> None:
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
seeds: [{seed}]
""".strip()
    )


def test_phase5_acceptance_covers_quick_eval_report_export_and_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    file_path = tmp_path / "cases.jsonl"
    file_path.write_text('{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n')
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root)

    fake_datasets_root = tmp_path / "fakepkgs" / "datasets"
    fake_datasets_root.mkdir(parents=True)
    (fake_datasets_root / "__init__.py").write_text(
        """
def load_dataset(dataset_name, *, split):
    return [{"id": "row-1", "prompt": {"question": "2+2"}, "answer": {"answer": "4"}}]
""".strip()
    )

    inline = _run_cli("quick-eval", "inline", "--input-json", '{"question":"2+2"}', "--expected-output-json", '{"answer":"4"}')
    file_eval = _run_cli("quick-eval", "file", "--path", str(file_path))
    benchmark = _run_cli("quick-eval", "benchmark", "--name", "mmlu_pro")
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
        env={"PYTHONPATH": f"{tmp_path / 'fakepkgs'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"},
    )
    run = _run_cli("run", "--config", str(config_path))
    resume = _run_cli("resume", "--config", str(config_path))
    report = _run_cli("report", "--config", str(config_path), "--format", "json")
    quickcheck = _run_cli("quickcheck", "--config", str(config_path))
    export_generation = _run_cli("export", "generation", "--config", str(config_path))

    for result in (inline, file_eval, benchmark, huggingface, run, resume, report, quickcheck, export_generation):
        assert result.returncode == 0, result.stderr

    assert json.loads(inline.stdout)["status"] == "completed"
    assert json.loads(file_eval.stdout)["metric_means"] == {"builtin/exact_match": 1.0}
    assert json.loads(benchmark.stdout)["status"] == "completed"
    assert json.loads(huggingface.stdout)["status"] == "completed"
    assert json.loads(run.stdout)["status"] == "completed"
    assert json.loads(resume.stdout)["status"] == "completed"
    assert json.loads(report.stdout)["run_result"]["status"] == "completed"
    assert json.loads(quickcheck.stdout)["metric_means"] == {"builtin/exact_match": 1.0}
    assert json.loads(export_generation.stdout)["run_id"] == json.loads(run.stdout)["run_id"]


def test_phase5_acceptance_covers_worker_pool_and_batch_execution(tmp_path: Path) -> None:
    worker_config = tmp_path / "worker.yaml"
    batch_config = tmp_path / "batch.yaml"
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    _write_config(worker_config, store_path=store_path, queue_root=queue_root, batch_root=batch_root, seed=7)
    _write_config(batch_config, store_path=store_path, queue_root=queue_root, batch_root=batch_root, seed=8)

    worker_submit = _run_cli("submit", "--config", str(worker_config), "--mode", "worker-pool")
    assert worker_submit.returncode == 0, worker_submit.stderr
    assert json.loads(_run_cli("resume", "--config", str(worker_config)).stdout)["status"] == "pending"
    worker_run = _run_cli("worker", "run", "--queue-root", str(queue_root), "--once")
    assert worker_run.returncode == 0, worker_run.stderr
    assert json.loads(worker_run.stdout)["status"] == "completed"

    batch_submit = _run_cli("submit", "--config", str(batch_config), "--mode", "batch")
    assert batch_submit.returncode == 0, batch_submit.stderr
    assert json.loads(_run_cli("resume", "--config", str(batch_config)).stdout)["status"] == "pending"
    batch_run = _run_cli("batch", "run", "--request", json.loads(batch_submit.stdout)["manifest_path"])
    assert batch_run.returncode == 0, batch_run.stderr
    assert json.loads(batch_run.stdout)["status"] == "completed"
