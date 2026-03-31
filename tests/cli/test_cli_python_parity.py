from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from themis import Reporter, StatsEngine, evaluate
from themis.core.base import JSONValue
from themis.core.config import StorageConfig
from themis.core.dataset_inputs import dataset_from_jsonl
from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.stores.factory import create_run_store


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "themis.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_config(path: Path, *, store_path: Path, queue_root: Path, batch_root: Path, seed: int | None) -> None:
    seeds_block = "" if seed is None else f"\nseeds: [{seed}]"
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
  - dataset_id: cases
    cases:
      - case_id: case-1
        input:
          question: 2+2
        expected_output:
          answer: "4"
{seeds_block}
""".strip()
    )


def test_python_api_and_cli_entrypoints_share_snapshot_identity_and_results(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text('{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n')
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    config_path = tmp_path / "experiment.yaml"
    _write_config(config_path, store_path=store_path, queue_root=queue_root, batch_root=batch_root, seed=None)

    experiment = Experiment.from_config(config_path)
    python_store = create_run_store(experiment.storage)
    python_store.initialize()
    python_result = experiment.run(store=python_store)
    python_benchmark = cast(dict[str, JSONValue], python_store.get_projection(python_result.run_id, "benchmark_result"))

    evaluate_result = evaluate(
        model="builtin/demo_generator",
        data=[dataset_from_jsonl(cases_path, dataset_id="cases")],
        metric="builtin/exact_match",
        parser="builtin/json_identity",
        storage=StorageConfig(store="memory"),
    )

    cli_run = _run_cli("run", "--config", str(config_path))
    cli_quick_eval = _run_cli("quick-eval", "file", "--path", str(cases_path))
    worker_submit = _run_cli("submit", "--config", str(config_path), "--mode", "worker-pool")
    worker_run = _run_cli("worker", "run", "--queue-root", str(queue_root))
    batch_submit = _run_cli("submit", "--config", str(config_path), "--mode", "batch")
    batch_manifest = json.loads(batch_submit.stdout)["manifest_path"]
    batch_run = _run_cli("batch", "run", "--request", batch_manifest)
    quickcheck = _run_cli("quickcheck", "--config", str(config_path))
    report = _run_cli("report", "--config", str(config_path), "--format", "json")

    assert cli_run.returncode == 0, cli_run.stderr
    assert cli_quick_eval.returncode == 0, cli_quick_eval.stderr
    assert worker_submit.returncode == 0, worker_submit.stderr
    assert worker_run.returncode == 0, worker_run.stderr
    assert batch_submit.returncode == 0, batch_submit.stderr
    assert batch_run.returncode == 0, batch_run.stderr
    assert quickcheck.returncode == 0, quickcheck.stderr
    assert report.returncode == 0, report.stderr

    cli_run_payload = json.loads(cli_run.stdout)
    cli_quick_eval_payload = json.loads(cli_quick_eval.stdout)
    worker_run_payload = json.loads(worker_run.stdout)
    batch_run_payload = json.loads(batch_run.stdout)
    quickcheck_payload = json.loads(quickcheck.stdout)
    report_payload = json.loads(report.stdout)

    assert python_result.run_id == evaluate_result.run_id
    assert python_result.run_id == cli_run_payload["run_id"] == cli_quick_eval_payload["run_id"]
    assert python_result.run_id == worker_run_payload["run_id"] == batch_run_payload["run_id"]
    assert python_benchmark["metric_means"] == cli_quick_eval_payload["metric_means"]
    assert quickcheck_payload["metric_means"] == python_benchmark["metric_means"]
    assert report_payload["benchmark_result"]["score_rows"] == Reporter(python_store).export_score_table(python_result.run_id)


def test_cli_compare_matches_python_stats_engine(tmp_path: Path) -> None:
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    _write_config(baseline_config, store_path=store_path, queue_root=queue_root, batch_root=batch_root, seed=7)
    _write_config(candidate_config, store_path=store_path, queue_root=queue_root, batch_root=batch_root, seed=8)

    baseline_experiment = Experiment.from_config(baseline_config)
    candidate_experiment = Experiment.from_config(candidate_config)
    store = create_run_store(baseline_experiment.storage)
    store.initialize()
    baseline_experiment.run(store=store)
    candidate_experiment.run(store=store)

    cli_compare = _run_cli(
        "compare",
        "--baseline-config",
        str(baseline_config),
        "--candidate-config",
        str(candidate_config),
    )

    assert cli_compare.returncode == 0, cli_compare.stderr
    cli_payload = json.loads(cli_compare.stdout)
    python_payload = StatsEngine().paired_compare(
        BenchmarkResult.model_validate(store.get_projection(baseline_experiment.compile().run_id, "benchmark_result")),
        BenchmarkResult.model_validate(store.get_projection(candidate_experiment.compile().run_id, "benchmark_result")),
    )

    assert cli_payload == python_payload
