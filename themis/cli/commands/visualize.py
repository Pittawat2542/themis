"""Visualization commands for interactive charts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.experiment.comparison import compare_experiments
from themis.experiment.visualization import (
    PLOTLY_AVAILABLE,
    InteractiveVisualizer,
    export_interactive_html,
)


def visualize_comparison_command(
    *,
    run_ids: Annotated[list[str], Parameter(help="Run IDs to visualize")],
    storage: Annotated[Path, Parameter(help="Storage directory")] = Path(".cache/runs"),
    metric: Annotated[str | None, Parameter(help="Metric to visualize")] = None,
    output: Annotated[Path, Parameter(help="Output HTML file path")] = Path(
        "visualization.html"
    ),
    chart_type: Annotated[
        str,
        Parameter(help="Chart type: comparison, evolution, dashboard, pareto"),
    ] = "comparison",
) -> int:
    """Generate interactive visualization for experiments.

    Examples:
        # Bar chart comparing accuracy across runs
        uv run python -m themis.cli visualize \\
          --run-ids run-1 run-2 run-3 \\
          --metric accuracy \\
          --output accuracy_comparison.html

        # Evolution chart showing metric over time
        uv run python -m themis.cli visualize \\
          --run-ids run-1 run-2 run-3 run-4 \\
          --metric accuracy \\
          --chart-type evolution \\
          --output accuracy_evolution.html

        # Dashboard with multiple metrics
        uv run python -m themis.cli visualize \\
          --run-ids run-1 run-2 run-3 \\
          --chart-type dashboard \\
          --output dashboard.html

        # Pareto frontier (requires --pareto-metrics and --maximize)
        uv run python -m themis.cli visualize-pareto \\
          --run-ids run-1 run-2 run-3 \\
          --metric1 accuracy \\
          --metric2 cost \\
          --output pareto.html
    """
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is not installed.")
        print("Install with: pip install plotly")
        return 1

    try:
        # Load experiments
        print(f"Loading experiments from {storage}...")
        comparison = compare_experiments(
            run_ids=run_ids,
            storage_dir=storage,
            include_metadata=True,
        )

        print(f"✓ Loaded {len(comparison.experiments)} experiments")

        # Create visualizer
        visualizer = InteractiveVisualizer()

        # Generate chart based on type
        if chart_type == "comparison":
            if not metric:
                metric = comparison.metrics[0] if comparison.metrics else "accuracy"
                print(f"Using default metric: {metric}")

            print(f"Creating comparison chart for '{metric}'...")
            fig = visualizer.plot_metric_comparison(comparison, metric)

        elif chart_type == "evolution":
            if not metric:
                metric = comparison.metrics[0] if comparison.metrics else "accuracy"
                print(f"Using default metric: {metric}")

            print(f"Creating evolution chart for '{metric}'...")
            fig = visualizer.plot_metric_evolution(comparison, metric)

        elif chart_type == "dashboard":
            print("Creating dashboard with multiple metrics...")
            fig = visualizer.create_dashboard(comparison)

        else:
            print(f"Error: Unknown chart type '{chart_type}'")
            print("Available: comparison, evolution, dashboard")
            return 1

        # Export to HTML
        export_interactive_html(fig, output)
        print(f"\n✓ Visualization saved to {output}")
        print("  Open in browser to interact with chart")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def visualize_pareto_command(
    *,
    run_ids: Annotated[list[str], Parameter(help="Run IDs to visualize")],
    storage: Annotated[Path, Parameter(help="Storage directory")] = Path(".cache/runs"),
    metric1: Annotated[str, Parameter(help="First metric (x-axis)")],
    metric2: Annotated[str, Parameter(help="Second metric (y-axis)")],
    maximize1: Annotated[bool, Parameter(help="Maximize metric1")] = True,
    maximize2: Annotated[bool, Parameter(help="Maximize metric2")] = True,
    output: Annotated[Path, Parameter(help="Output HTML file path")] = Path(
        "pareto.html"
    ),
) -> int:
    """Generate Pareto frontier visualization.

    Examples:
        # Maximize accuracy, minimize cost
        uv run python -m themis.cli visualize-pareto \\
          --run-ids run-1 run-2 run-3 run-4 \\
          --metric1 accuracy \\
          --metric2 cost \\
          --maximize1 true \\
          --maximize2 false \\
          --output pareto.html
    """
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is not installed.")
        print("Install with: pip install plotly")
        return 1

    try:
        # Load experiments
        print(f"Loading experiments from {storage}...")
        comparison = compare_experiments(
            run_ids=run_ids,
            storage_dir=storage,
            include_metadata=True,
        )

        print(f"✓ Loaded {len(comparison.experiments)} experiments")

        # Compute Pareto frontier
        print(f"Computing Pareto frontier for {metric1} and {metric2}...")
        pareto_ids = comparison.pareto_frontier(
            objectives=[metric1, metric2],
            maximize=[maximize1, maximize2],
        )

        print(f"✓ Found {len(pareto_ids)} Pareto-optimal experiments:")
        for run_id in pareto_ids:
            print(f"  - {run_id}")

        # Create visualization
        visualizer = InteractiveVisualizer()
        fig = visualizer.plot_pareto_frontier(
            comparison, metric1, metric2, pareto_ids, maximize1, maximize2
        )

        # Export to HTML
        export_interactive_html(fig, output)
        print(f"\n✓ Visualization saved to {output}")
        print("  Red points are Pareto-optimal")
        print("  Blue points are dominated")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def visualize_distribution_command(
    *,
    run_id: Annotated[str, Parameter(help="Run ID to visualize")],
    storage: Annotated[Path, Parameter(help="Storage directory")] = Path(".cache/runs"),
    metric: Annotated[str, Parameter(help="Metric to visualize")],
    plot_type: Annotated[
        str, Parameter(help="Plot type: histogram, box, violin")
    ] = "histogram",
    output: Annotated[Path, Parameter(help="Output HTML file path")] = Path(
        "distribution.html"
    ),
) -> int:
    """Generate metric distribution visualization.

    Shows the distribution of a metric across all samples in an experiment.

    Examples:
        # Histogram of accuracy scores
        uv run python -m themis.cli visualize-distribution \\
          --run-id my-run \\
          --metric accuracy \\
          --output accuracy_dist.html

        # Violin plot
        uv run python -m themis.cli visualize-distribution \\
          --run-id my-run \\
          --metric accuracy \\
          --plot-type violin \\
          --output accuracy_violin.html
    """
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is not installed.")
        print("Install with: pip install plotly")
        return 1

    try:
        import json

        # Load report
        print(f"Loading report from {storage / run_id}...")
        report_path = storage / run_id / "report.json"

        if not report_path.exists():
            print(f"Error: Report not found at {report_path}")
            return 1

        with report_path.open("r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Extract evaluation report
        # Note: This is simplified - in production you'd deserialize properly
        from themis.core.entities import EvaluationRecord, MetricScore
        from themis.evaluation.reports import EvaluationReport, MetricAggregate

        # Build evaluation report from JSON
        records = []
        for sample_data in report_data.get("samples", []):
            scores = [
                MetricScore(
                    metric_name=score["metric"],
                    value=score["value"],
                    details=score.get("details"),
                    metadata=score.get("metadata", {}),
                )
                for score in sample_data["scores"]
            ]
            records.append(
                EvaluationRecord(
                    sample_id=sample_data["sample_id"],
                    scores=scores,
                    failures=[],
                )
            )

        # Build metric aggregates
        metrics = {}
        for metric_data in report_data.get("metrics", []):
            metrics[metric_data["name"]] = MetricAggregate(
                count=metric_data["count"],
                mean=metric_data["mean"],
            )

        eval_report = EvaluationReport(
            records=records,
            metrics=metrics,
            failures=[],
        )

        print(f"✓ Loaded report with {len(records)} samples")

        # Create visualization
        visualizer = InteractiveVisualizer()
        fig = visualizer.plot_metric_distribution(eval_report, metric, plot_type)

        # Export to HTML
        export_interactive_html(fig, output)
        print(f"\n✓ Visualization saved to {output}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


__all__ = [
    "visualize_comparison_command",
    "visualize_pareto_command",
    "visualize_distribution_command",
]
