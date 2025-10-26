
from experiments.agentic_example import cli as agentic_cli
from experiments.agentic_example import config as agentic_config
from experiments.agentic_example import experiment as agentic_experiment


def test_agentic_experiment_runs(tmp_path):
    config = agentic_config.AGENTIC_DEFAULT_CONFIG.apply_overrides(
        run_id="agentic-test",
        storage_dir=tmp_path / "storage",
    )
    report = agentic_experiment.run_experiment(config)

    assert report.metadata["total_samples"] == 2
    assert set(report.evaluation_report.metrics.keys()) == {
        "ExactMatch",
        "ResponseLength",
    }


def test_agentic_cli_dry_run(tmp_path, capsys):
    config = agentic_config.AGENTIC_DEFAULT_CONFIG.apply_overrides(
        run_id="agentic-cli",
        storage_dir=tmp_path / "storage",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    exit_code = agentic_cli.main(
        [
            "run",
            "--config-path",
            str(config_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert "Planner prompt" in capsys.readouterr().out
