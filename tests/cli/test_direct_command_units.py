from __future__ import annotations

import json
from pathlib import Path

from themis.cli.commands.batch import run as run_batch_command
from themis.cli.commands.compare import compare
from themis.cli.commands.export import evaluation as export_evaluation
from themis.cli.commands.export import generation as export_generation
from themis.cli.commands.init import init
from themis.cli.commands.inspect import evaluation as inspect_evaluation
from themis.cli.commands.inspect import case as inspect_case
from themis.cli.commands.inspect import lineage as inspect_lineage
from themis.cli.commands.inspect import run_record as inspect_run_record
from themis.cli.commands.inspect import runs as inspect_runs
from themis.cli.commands.inspect import snapshot as inspect_snapshot
from themis.cli.commands.inspect import state as inspect_state
from themis.cli.commands.inspect import telemetry as inspect_telemetry
from themis.cli.commands.quick_eval import benchmark, file as quick_eval_file
from themis.cli.commands.quick_eval import inline as quick_eval_inline
from themis.cli.commands.reporting import report
from themis.cli.commands.run import estimate, quickcheck, replay, resume, run
from themis.cli.commands.worker import run as run_worker_command
from themis.core.experiment import Experiment
from themis.core.read_models import BenchmarkResult
from themis.core.registry import RunLineage
from themis.core.results import RunStatus
from themis.core.stores.factory import create_run_store


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
        == "metric_id,count,mean,min,max,ci_lower,ci_upper"
    )

    assert report(config=str(baseline_config), format="latex") == 0
    assert "\\begin{tabular}" in capsys.readouterr().out

    assert report(config=str(baseline_config), run_id=baseline_result.run_id) == 0
    report_by_id_payload = json.loads(capsys.readouterr().out)
    assert report_by_id_payload["run_result"]["run_id"] == baseline_result.run_id

    baseline_store = create_run_store(baseline_experiment.storage)
    baseline_store.initialize()
    baseline_store.update_run_record(baseline_result.run_id, baseline_label="main")

    assert report(config=str(baseline_config), baseline_label="main") == 0
    report_by_label_payload = json.loads(capsys.readouterr().out)
    assert report_by_label_payload["run_result"]["run_id"] == baseline_result.run_id

    assert (
        compare(
            baseline_config=str(baseline_config),
            candidate_config=str(candidate_config),
        )
        == 0
    )
    compare_payload = json.loads(capsys.readouterr().out)
    assert compare_payload["metrics"][0]["metric_id"] == "builtin/exact_match"
    assert compare_payload["metrics"][0]["ties"] == 1

    assert (
        compare(
            baseline_config=str(baseline_config),
            candidate_config=str(candidate_config),
            baseline_run_id=baseline_result.run_id,
            candidate_run_id=candidate_experiment.compile().run_id,
        )
        == 0
    )
    compare_by_id_payload = json.loads(capsys.readouterr().out)
    assert compare_by_id_payload["metrics"][0]["metric_id"] == "builtin/exact_match"
    assert compare_by_id_payload["metrics"][0]["ties"] == 1

    candidate_store.update_run_record(
        candidate_experiment.compile().run_id, baseline_label="candidate"
    )
    assert (
        compare(
            baseline_config=str(baseline_config),
            candidate_config=str(candidate_config),
            baseline_baseline_label="main",
            candidate_baseline_label="candidate",
        )
        == 0
    )
    compare_by_label_payload = json.loads(capsys.readouterr().out)
    assert compare_by_label_payload["metrics"][0]["metric_id"] == "builtin/exact_match"
    assert compare_by_label_payload["metrics"][0]["ties"] == 1

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
    experiment, store, result = run_config_experiment(config_path)
    store.update_run_record(
        result.run_id,
        tags=["phase1", "smoke"],
        baseline_label="main",
    )

    assert inspect_snapshot(config=str(config_path)) == 0
    snapshot_payload = json.loads(capsys.readouterr().out)
    assert snapshot_payload["run_id"] == result.run_id

    assert inspect_state(config=str(config_path)) == 0
    state_payload = json.loads(capsys.readouterr().out)
    assert state_payload["status"] == "completed"

    assert inspect_runs(config=str(config_path), tag=["phase1"]) == 0
    run_list_payload = json.loads(capsys.readouterr().out)
    assert run_list_payload[0]["run_id"] == result.run_id

    assert inspect_run_record(config=str(config_path), run_id=result.run_id) == 0
    run_record_payload = json.loads(capsys.readouterr().out)
    assert run_record_payload["baseline_label"] == "main"

    store.update_run_record(
        result.run_id,
        lineage=[RunLineage(parent_run_id="parent-run", relationship="rerun")],
    )
    assert inspect_lineage(config=str(config_path), run_id=result.run_id) == 0
    lineage_payload = json.loads(capsys.readouterr().out)
    assert lineage_payload["run_id"] == result.run_id
    assert lineage_payload["lineage"][0]["parent_run_id"] == "parent-run"

    assert (
        inspect_case(
            config=str(config_path),
            run_id=result.run_id,
            case_id="case-1",
            dataset_id="cases",
        )
        == 0
    )
    case_payload = json.loads(capsys.readouterr().out)
    assert case_payload["case_id"] == "case-1"

    assert inspect_telemetry(config=str(config_path), run_id=result.run_id) == 0
    telemetry_payload = json.loads(capsys.readouterr().out)
    assert telemetry_payload["run_id"] == result.run_id

    assert replay(config=str(config_path), stage="score") == 0
    replay_payload = json.loads(capsys.readouterr().out)
    assert replay_payload["status"] == "completed"

    try:
        inspect_evaluation(
            config=str(config_path),
            case_id="case-1",
            metric_id="builtin/llm_rubric",
            dataset_id="cases",
        )
    except SystemExit as exc:
        assert "No evaluation execution found" in str(exc)
        assert "dataset_id=cases" in str(exc)
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
