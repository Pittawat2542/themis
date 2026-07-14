from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from themis.analysis import Reporter, StatsEngine
from themis.core.base import JSONValue
from themis.core.read_models import BenchmarkResult
from themis.core.stores.factory import create_run_store
from themis.launcher import _load_runtime_experiment
from tests.cli.helpers import run_cli


pytestmark = pytest.mark.slow


def _cli_data(output: str):
    envelope = json.loads(output)
    assert envelope["schema_version"] == "1"
    assert isinstance(envelope["command"], str)
    return envelope["data"]


def _write_config(
    path: Path,
    *,
    store_path: Path,
    queue_root: Path,
    batch_root: Path,
    seed: int | None,
) -> None:
    definition_module = f"{path.stem}_definition"
    seeds = [] if seed is None else [seed]
    path.with_name(f"{definition_module}.py").write_text(
        f"""from themis import Case, Dataset, Evaluation, Experiment, Generation

experiment = Experiment(
    datasets=[Dataset(
        dataset_id="cases",
        cases=[Case(
            case_id="case-1",
            input={{"question": "2+2"}},
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
    seeds={seeds!r},
)
"""
    )
    path.write_text(
        f"""
definition: {definition_module}:experiment
storage:
  target: sqlite
  kwargs:
    path: {store_path}
runtime:
  queue_root: {queue_root}
  batch_root: {batch_root}
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

    experiment = _load_runtime_experiment(config_path)
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
    worker_run = run_cli(
        "worker",
        "run",
        "--queue-root",
        str(queue_root),
        "--definition-root",
        str(tmp_path),
    )
    batch_submit = run_cli("submit", "--config", str(config_path), "--mode", "batch")
    batch_manifest = _cli_data(batch_submit.stdout)["manifest_path"]
    batch_run = run_cli(
        "batch",
        "run",
        "--request",
        batch_manifest,
        "--definition-root",
        str(tmp_path),
    )
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

    cli_run_payload = _cli_data(cli_run.stdout)
    cli_quick_eval_payload = _cli_data(cli_quick_eval.stdout)
    worker_run_payload = _cli_data(worker_run.stdout)
    batch_run_payload = _cli_data(batch_run.stdout)
    quickcheck_payload = _cli_data(quickcheck.stdout)
    report_payload = _cli_data(report.stdout)

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

    baseline_experiment = _load_runtime_experiment(baseline_config)
    candidate_experiment = _load_runtime_experiment(candidate_config)
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
    cli_payload = _cli_data(cli_compare.stdout)
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
