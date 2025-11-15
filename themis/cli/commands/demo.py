"""Demo command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.cli.utils import effective_total, export_outputs
from themis.experiment import math as math_experiment
from themis.utils.logging_utils import configure_logging
from themis.utils.progress import ProgressReporter


def demo_command(
    *,
    max_samples: Annotated[
        int | None, Parameter(help="Limit number of demo samples")
    ] = None,
    log_level: Annotated[
        str, Parameter(help="Logging level (critical/error/warning/info/debug/trace)")
    ] = "info",
    csv_output: Annotated[
        Path | None, Parameter(help="Write CSV export to this path")
    ] = None,
    html_output: Annotated[
        Path | None, Parameter(help="Write HTML summary to this path")
    ] = None,
    json_output: Annotated[
        Path | None, Parameter(help="Write JSON export to this path")
    ] = None,
) -> int:
    """Run the built-in demo dataset."""
    configure_logging(log_level)
    dataset = [
        {
            "unique_id": "demo-1",
            "problem": "Convert the point (0,3) in rectangular coordinates to polar coordinates.",
            "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
            "subject": "precalculus",
            "level": 2,
        },
        {
            "unique_id": "demo-2",
            "problem": "What is 7 + 5?",
            "answer": "12",
            "subject": "arithmetic",
            "level": 1,
        },
    ]
    experiment = math_experiment.build_math500_zero_shot_experiment()
    total = effective_total(len(dataset), max_samples)
    with ProgressReporter(total=total, description="Generating") as progress:
        report = experiment.run(
            dataset,
            max_samples=max_samples,
            on_result=progress.on_result,
        )
    print(math_experiment.summarize_report(report))
    export_outputs(
        report,
        csv_output=csv_output,
        html_output=html_output,
        json_output=json_output,
        title="Demo experiment",
    )
    return 0
