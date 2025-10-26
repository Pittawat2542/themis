
from experiments.example import cli as example_cli
from experiments.example import config as example_config
from experiments.example import experiment as example_experiment


def test_run_experiment_with_demo(tmp_path):
    config = example_config.DEFAULT_CONFIG.apply_overrides(
        run_id="test-demo",
        storage_dir=tmp_path / "storage",
    )

    report = example_experiment.run_experiment(config)

    assert report.metadata["total_samples"] == 2
    assert report.evaluation_report.metrics["ExactMatch"].mean == 1.0
    assert len(report.evaluation_report.records) == 2


def test_example_cli_runs_full_flow(tmp_path, capsys):
    config = example_config.DEFAULT_CONFIG.apply_overrides(
        run_id="cli-demo",
        storage_dir=tmp_path / "storage",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    exit_code = example_cli.main(
        [
            "run",
            "--config-path",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "exact match" in captured.out.lower()
