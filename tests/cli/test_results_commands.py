from __future__ import annotations

import json
from pathlib import Path

import themis.cli.commands.results as results


def _write_summary(run_dir: Path, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "total_samples": 2,
        "metrics": {"ExactMatch": {"mean": 0.5, "count": 2}},
        "metadata": {"model": "fake:fake-math-llm"},
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_results_list_reads_default_experiments_layout(tmp_path, capsys):
    storage = tmp_path / ".cache" / "experiments"
    run_dir = storage / "experiments" / "default" / "runs" / "run-1"
    _write_summary(run_dir, "run-1")

    exit_code = results.list_command(storage=storage)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "run-1" in output


def test_results_commands_fallback_to_legacy_storage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    legacy_run_dir = Path(".cache/runs/run-legacy")
    _write_summary(legacy_run_dir, "run-legacy")

    exit_code = results.summary_command(run_id="run-legacy")
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Warning: Falling back to legacy storage path" in output
    assert "run-legacy" in output
