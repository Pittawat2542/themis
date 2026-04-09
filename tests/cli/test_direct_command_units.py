from __future__ import annotations

import json
from pathlib import Path

from themis.cli.commands.batch import run as run_batch_command
from themis.cli.commands.compare import compare
from themis.cli.commands.export import evaluation as export_evaluation
from themis.cli.commands.export import generation as export_generation
from themis.cli.commands.init import init
from themis.cli.commands.inspect import evaluation as inspect_evaluation
from themis.cli.commands.inspect import snapshot as inspect_snapshot
from themis.cli.commands.inspect import state as inspect_state
from themis.cli.commands.quick_eval import benchmark, file as quick_eval_file
from themis.cli.commands.quick_eval import inline as quick_eval_inline
from themis.cli.commands.reporting import report
from themis.cli.commands.run import estimate, quickcheck, replay, resume, run
from themis.cli.commands.worker import run as run_worker_command
from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.results import RunStatus


def test_run_resume_estimate_and_quickcheck_commands(
    write_experiment_config, capsys
) -> None:
    config_path = write_experiment_config()

    assert run(config=str(config_path)) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "completed"
    assert run_payload["metric_means"]["builtin/exact_match"] == 1.0

    assert resume(config=str(config_path)) == 0
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_payload["completed_cases"] == 1

    assert estimate(config=str(config_path)) == 0
    estimate_payload = json.loads(capsys.readouterr().out)
    assert estimate_payload["planned_generation_tasks"] >= 1

    assert quickcheck(config=str(config_path)) == 0
    quickcheck_payload = json.loads(capsys.readouterr().out)
    assert quickcheck_payload["status"] == "completed"


def test_report_export_and_compare_commands(
    write_experiment_config, run_config_experiment, capsys
) -> None:
    baseline_config = write_experiment_config(name="baseline.yaml", answer="4", seed=7)
    candidate_config = write_experiment_config(
        name="candidate.yaml", answer="4", seed=8
    )

    baseline_experiment, _, baseline_result = run_config_experiment(baseline_config)
    candidate_experiment, candidate_store, _ = run_config_experiment(candidate_config)

    assert report(config=str(baseline_config), format="json") == 0
    report_payload = json.loads(capsys.readouterr().out)
    assert report_payload["run_result"]["run_id"] == baseline_result.run_id

    assert report(config=str(baseline_config), format="markdown") == 0
    assert "# Run Report" in capsys.readouterr().out

    assert report(config=str(baseline_config), format="csv") == 0
    assert (
        capsys.readouterr().out.splitlines()[0]
        == "case_id,dataset_id,case_key,metric_id,outcome,value,candidate_id,error_category,error_message,details"
    )

    assert report(config=str(baseline_config), format="latex") == 0
    assert "\\begin{tabular}" in capsys.readouterr().out

    assert (
        compare(
            baseline_config=str(baseline_config),
            candidate_config=str(candidate_config),
        )
        == 0
    )
    compare_payload = json.loads(capsys.readouterr().out)
    assert compare_payload["metrics"]["builtin/exact_match"]["ties"] == 1

    assert export_generation(config=str(baseline_config)) == 0
    generation_payload = json.loads(capsys.readouterr().out)
    assert generation_payload["run_id"] == baseline_result.run_id

    assert export_evaluation(config=str(baseline_config)) == 0
    evaluation_payload = json.loads(capsys.readouterr().out)
    assert evaluation_payload["run_id"] == baseline_result.run_id

    benchmark_projection = candidate_store.get_projection(
        candidate_experiment.compile().run_id, "benchmark_result"
    )
    assert BenchmarkResult.model_validate(benchmark_projection).run_id == (
        candidate_experiment.compile().run_id
    )


def test_inspect_commands_and_replay_command(
    write_experiment_config, run_config_experiment, capsys
) -> None:
    config_path = write_experiment_config()
    experiment, _, result = run_config_experiment(config_path)

    assert inspect_snapshot(config=str(config_path)) == 0
    snapshot_payload = json.loads(capsys.readouterr().out)
    assert snapshot_payload["run_id"] == result.run_id

    assert inspect_state(config=str(config_path)) == 0
    state_payload = json.loads(capsys.readouterr().out)
    assert state_payload["status"] == "completed"

    assert replay(config=str(config_path), stage="score") == 0
    replay_payload = json.loads(capsys.readouterr().out)
    assert replay_payload["status"] == "completed"

    try:
        inspect_evaluation(
            config=str(config_path),
            case_id="case-1",
            metric_id="builtin/llm_rubric",
        )
    except SystemExit as exc:
        assert "No evaluation execution found" in str(exc)
    else:
        raise AssertionError("expected inspect.evaluation to report missing execution")

    assert (
        Experiment.from_config(config_path).compile().run_id
        == experiment.compile().run_id
    )


def test_quick_eval_and_init_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"case_id":"case-1","input":{"question":"2+2"},"expected_output":{"answer":"4"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "themis.catalog.benchmarks.materializers.load_huggingface_rows",
        lambda dataset_id, split, revision=None, config_name=None: [
            {
                "item_id": "mmlu-pro-1",
                "question": "Which planet is known as the Red Planet?",
                "options": ["Venus", "Mars", "Jupiter", "Mercury"],
                "answer": "B",
                "category": "astronomy",
                "src": "fixture",
            }
        ],
    )

    assert (
        quick_eval_inline(
            input_json='{"question":"2+2"}',
            expected_output_json='{"answer":"4"}',
        )
        == 0
    )
    inline_payload = json.loads(capsys.readouterr().out)
    assert inline_payload["status"] == "completed"

    assert quick_eval_file(path=str(jsonl_path)) == 0
    file_payload = json.loads(capsys.readouterr().out)
    assert file_payload["metric_means"]["builtin/exact_match"] == 1.0

    assert benchmark(name="mmlu_pro") == 0
    benchmark_payload = json.loads(capsys.readouterr().out)
    assert benchmark_payload["run_id"]

    project_root = tmp_path / "scaffold"
    assert init(path=str(project_root)) == 0
    init_output = capsys.readouterr().out.strip()
    assert init_output == str(project_root)
    assert (project_root / "experiment.yaml").is_file()
    assert (project_root / "data" / "sample.jsonl").is_file()
    assert (project_root / "run.py").is_file()


def test_worker_and_batch_commands_serialize_results(monkeypatch, capsys) -> None:
    class _Result:
        def __init__(self, run_id: str, status: RunStatus) -> None:
            self.run_id = run_id
            self.status = status

    monkeypatch.setattr(
        "themis.cli.commands.worker.run_worker_once",
        lambda queue_root: _Result("run-1", RunStatus.COMPLETED),
    )
    assert run_worker_command(queue_root="queue") == 0
    worker_payload = json.loads(capsys.readouterr().out)
    assert worker_payload == {"run_id": "run-1", "status": "completed"}

    monkeypatch.setattr(
        "themis.cli.commands.worker.run_worker_once",
        lambda queue_root: None,
    )
    assert run_worker_command(queue_root="queue") == 0
    assert json.loads(capsys.readouterr().out) == {"status": "idle"}

    monkeypatch.setattr(
        "themis.cli.commands.batch.run_batch_request",
        lambda request: _Result("run-2", RunStatus.COMPLETED),
    )
    assert run_batch_command(request="request.json") == 0
    batch_payload = json.loads(capsys.readouterr().out)
    assert batch_payload == {"run_id": "run-2", "status": "completed"}
