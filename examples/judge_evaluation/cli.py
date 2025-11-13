"""CLI for running judge evaluation experiments."""

from __future__ import annotations

import cyclopts

from . import experiment
from .config import DEFAULT_JUDGE_CONFIG

app = cyclopts.App(name="judge-evaluation", help="Judge-based evaluation experiments")


@app.default
def run():
    """Run the judge evaluation experiment."""
    print("Running judge evaluation experiment...")
    report = experiment.run_experiment(DEFAULT_JUDGE_CONFIG)
    summary = experiment.summarize_report(report)
    print(f"\n{summary}\n")


if __name__ == "__main__":
    app()
