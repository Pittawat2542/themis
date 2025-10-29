from __future__ import annotations

from pathlib import Path

from cyclopts import App

from themis.config import load_experiment_config, run_experiment_from_config

app = App(help="Run experiments for the {{project_name}} project.")


@app.command()
def run(
    config_path: Path = Path("config.sample.json"),
) -> int:
    config = load_experiment_config(config_path)
    report = run_experiment_from_config(config)
    print(report.metadata)
    return 0


if __name__ == "__main__":
    app()
