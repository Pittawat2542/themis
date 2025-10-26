
from experiments.advanced_example import cli as advanced_cli
from experiments.advanced_example import config as advanced_config
from experiments.advanced_example import experiment as advanced_experiment


def test_advanced_experiment_runs_and_tracks_subjects(tmp_path):
    config = advanced_config.ADVANCED_DEFAULT_CONFIG.apply_overrides(
        run_id="advanced-test",
        storage_dir=tmp_path / "storage",
    )
    report = advanced_experiment.run_experiment(config)

    assert report.metadata["total_samples"] == 2
    breakdown = report.metadata.get("subject_breakdown")
    assert breakdown is not None
    assert "precalculus" in breakdown or "arithmetic" in breakdown
    call_history = report.metadata.get("generation_call_history")
    assert call_history is not None
    assert call_history == sorted(call_history)
    for record in report.generation_results:
        assert record.metrics["attempt_count"] == config.test_time_attempts


def test_advanced_cli_with_config(tmp_path, capsys):
    config = advanced_config.ADVANCED_DEFAULT_CONFIG.apply_overrides(
        run_id="advanced-cli",
        storage_dir=tmp_path / "storage",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    exit_code = advanced_cli.main(
        [
            "run",
            "--config-path",
            str(config_path),
            "--prompt-style",
            "concise",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SubjectBreakdown" in captured.out
