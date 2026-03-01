"""Metric visualizations (bar charts, distributions, evolution series)."""

from __future__ import annotations

from themis.evaluation.reports import EvaluationReport
from themis.exceptions import ConfigurationError, MetricError
from themis.experiment.comparison import MultiExperimentComparison
from themis.experiment.visualization.core import _check_plotly, go


def plot_metric_comparison(
    comparison: MultiExperimentComparison,
    metric: str,
    title: str | None = None,
    show_values: bool = True,
) -> "go.Figure":
    """Create bar chart comparing metric across experiments.

    Args:
        comparison: Multi-experiment comparison
        metric: Metric name to visualize
        title: Chart title (default: "{metric} Comparison")
        show_values: Show values on bars

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    if metric not in comparison.metrics and metric not in ("cost", "total_cost"):
        raise MetricError(
            f"Metric '{metric}' not found. Available: {comparison.metrics}"
        )

    # Extract data
    run_ids = [exp.run_id for exp in comparison.experiments]
    values = [exp.get_metric(metric) or 0.0 for exp in comparison.experiments]

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=run_ids,
                y=values,
                text=[f"{v:.4f}" for v in values] if show_values else None,
                textposition="auto",
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<br><extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title or f"{metric} Comparison",
        xaxis_title="Run ID",
        yaxis_title=metric,
        hovermode="x unified",
        template="plotly_white",
        font=dict(size=12),
    )

    return fig


def plot_metric_distribution(
    report: EvaluationReport,
    metric: str,
    plot_type: str = "histogram",
    title: str | None = None,
) -> "go.Figure":
    """Create histogram or violin plot of metric distribution.

    Args:
        report: Evaluation report
        metric: Metric name
        plot_type: "histogram", "box", or "violin"
        title: Chart title

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    if metric not in report.metrics:
        raise MetricError(
            f"Metric '{metric}' not found. Available: {list(report.metrics.keys())}"
        )

    # Extract metric values per sample
    values = []
    for record in report.records:
        for score in record.scores:
            if score.metric_name == metric:
                values.append(score.value)

    if not values:
        raise MetricError(f"No values found for metric '{metric}'")

    # Create plot based on type
    if plot_type == "histogram":
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=values,
                    nbinsx=30,
                    hovertemplate="Value: %{x:.4f}<br>Count: %{y}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            xaxis_title=metric,
            yaxis_title="Count",
        )
    elif plot_type == "box":
        fig = go.Figure(
            data=[
                go.Box(
                    y=values,
                    name=metric,
                    boxmean="sd",
                    hovertemplate="Value: %{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(yaxis_title=metric)
    elif plot_type == "violin":
        fig = go.Figure(
            data=[
                go.Violin(
                    y=values,
                    name=metric,
                    box_visible=True,
                    meanline_visible=True,
                    hovertemplate="Value: %{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(yaxis_title=metric)
    else:
        raise ConfigurationError(
            f"Unknown plot_type '{plot_type}'. Use 'histogram', 'box', or 'violin'"
        )

    fig.update_layout(
        title=title or f"{metric} Distribution ({len(values)} samples)",
        template="plotly_white",
        font=dict(size=12),
    )

    return fig


def plot_metric_evolution(
    comparison: MultiExperimentComparison,
    metric: str,
    title: str | None = None,
) -> "go.Figure":
    """Create line plot showing metric evolution across runs.

    Experiments are ordered by timestamp if available.

    Args:
        comparison: Multi-experiment comparison
        metric: Metric name
        title: Chart title

    Returns:
        Plotly Figure object
    """
    _check_plotly()
    if metric not in comparison.metrics and metric not in ("cost", "total_cost"):
        raise MetricError(
            f"Metric '{metric}' not found. Available: {comparison.metrics}"
        )

    # Sort experiments by timestamp if available
    sorted_exps = sorted(
        comparison.experiments,
        key=lambda e: e.timestamp or "",
    )

    # Extract data
    x_labels = [exp.run_id for exp in sorted_exps]
    y_values = [exp.get_metric(metric) or 0.0 for exp in sorted_exps]

    # Create line chart
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_labels,
                y=y_values,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>"
                + f"{metric}: %{{y:.4f}}<br>"
                + "<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title or f"{metric} Evolution Over Time",
        xaxis_title="Run ID (chronological)",
        yaxis_title=metric,
        template="plotly_white",
        font=dict(size=12),
        hovermode="x unified",
    )

    return fig
