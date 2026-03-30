"""Reporting CLI commands."""

from __future__ import annotations

from typing import Literal

from themis.cli.helpers import initialize_store, load_experiment
from themis.core.reporter import Reporter


def report(*, config: str, format: Literal["json", "markdown", "csv", "latex"] = "json") -> int:
    experiment = load_experiment(config)
    snapshot = experiment.compile()
    store = initialize_store(experiment)
    reporter = Reporter(store)
    if format == "json":
        print(reporter.export_json(snapshot.run_id))
        return 0
    if format == "markdown":
        print(reporter.export_markdown(snapshot.run_id))
        return 0
    if format == "csv":
        print(reporter.export_csv(snapshot.run_id))
        return 0
    print(reporter.export_latex(snapshot.run_id))
    return 0
