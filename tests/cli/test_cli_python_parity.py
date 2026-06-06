from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from themis import Reporter, StatsEngine
from themis.core.base import JSONValue
from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.stores.factory import create_run_store
from tests.cli.helpers import run_cli


pytestmark = pytest.mark.slow


def _write_config(
    path: Path,
    *,
    store_path: Path,
    queue_root: Path,
    batch_root: Path,
    seed: int | None,
) -> None:
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
  target: sqlite
  kwargs:
    path: {store_path}
runtime:
  queue_root: {queue_root}
  batch_root: {batch_root}
dataset_sources:
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


def test_python_api_and_cli_entrypoints_share_snapshot_identity_and_results(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n'
    )
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    config_path = tmp_path / "experiment.yaml"
    _write_config(
        config_path,
        store_path=store_path,
        queue_root=queue_root,
        batch_root=batch_root,
        seed=None,
    )

    experiment = Experiment.from_config(config_path)
    python_store = create_run_store(experiment.storage)
    python_store.initialize()
    python_result = experiment.run(store=python_store)
    python_benchmark = cast(
        dict[str, JSONValue],
        python_store.get_projection(python_result.run_id, "benchmark_result"),
    )

    cli_run = run_cli("run", "--config", str(config_path))
    cli_quick_eval = run_cli("quick-eval", "file", "--path", str(cases_path))
    worker_submit = run_cli(
        "submit", "--config", str(config_path), "--mode", "worker-pool"
    )
    worker_run = run_cli("worker", "run", "--queue-root", str(queue_root))
    batch_submit = run_cli("submit", "--config", str(config_path), "--mode", "batch")
    batch_manifest = json.loads(batch_submit.stdout)["manifest_path"]
    batch_run = run_cli("batch", "run", "--request", batch_manifest)
    quickcheck = run_cli("quickcheck", "--config", str(config_path))
    report = run_cli("report", "--config", str(config_path), "--format", "json")

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

    assert (
        python_result.run_id
        == cli_run_payload["run_id"]
        == cli_quick_eval_payload["run_id"]
    )
    assert (
        python_result.run_id
        == worker_run_payload["run_id"]
        == batch_run_payload["run_id"]
    )
    assert python_benchmark["metric_means"] == cli_quick_eval_payload["metric_means"]
    assert quickcheck_payload["metric_means"] == python_benchmark["metric_means"]
    assert report_payload["stats_summary"] == Reporter(python_store).summary(
        python_result.run_id
    ).model_dump(mode="json")


def test_cli_compare_matches_python_stats_engine(tmp_path: Path) -> None:
    store_path = tmp_path / "runs.sqlite3"
    queue_root = tmp_path / "queue"
    batch_root = tmp_path / "batch"
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    _write_config(
        baseline_config,
        store_path=store_path,
        queue_root=queue_root,
        batch_root=batch_root,
        seed=7,
    )
    _write_config(
        candidate_config,
        store_path=store_path,
        queue_root=queue_root,
        batch_root=batch_root,
        seed=8,
    )

    baseline_experiment = Experiment.from_config(baseline_config)
    candidate_experiment = Experiment.from_config(candidate_config)
    store = create_run_store(baseline_experiment.storage)
    store.initialize()
    baseline_experiment.run(store=store)
    candidate_experiment.run(store=store)

    cli_compare = run_cli(
        "compare",
        "--baseline-config",
        str(baseline_config),
        "--candidate-config",
        str(candidate_config),
    )

    assert cli_compare.returncode == 0, cli_compare.stderr
    cli_payload = json.loads(cli_compare.stdout)
    python_payload = (
        StatsEngine()
        .compare(
            BenchmarkResult.model_validate(
                store.get_projection(
                    baseline_experiment.compile().run_id, "benchmark_result"
                )
            ),
            BenchmarkResult.model_validate(
                store.get_projection(
                    candidate_experiment.compile().run_id, "benchmark_result"
                )
            ),
        )
        .model_dump(mode="json")
    )

    assert cli_payload == python_payload
