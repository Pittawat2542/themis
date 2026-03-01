"""Dashboard visualizations."""

from __future__ import annotations

from themis.experiment.comparison import MultiExperimentComparison
from themis.experiment.visualization.core import _check_plotly, go, make_subplots


def create_dashboard(
    comparison: MultiExperimentComparison,
    metrics: list[str] | None = None,
    include_cost: bool = True,
) -> "go.Figure":
    """Create comprehensive dashboard with multiple charts.

    Args:
        comparison: Multi-experiment comparison
        metrics: Metrics to visualize (default: all)
        include_cost: Include cost visualization if available

    Returns:
        Plotly Figure with subplots
    """
    _check_plotly()
    metrics_to_plot = metrics or comparison.metrics[:4]  # Limit to 4 for layout

    # Check if cost data is available
    has_cost = include_cost and any(
        exp.get_cost() is not None for exp in comparison.experiments
    )

    # Determine subplot layout
    n_metrics = len(metrics_to_plot)
    n_plots = n_metrics + (1 if has_cost else 0)

    rows = (n_plots + 1) // 2  # 2 columns
    cols = 2 if n_plots > 1 else 1

    # Create subplots
    subplot_titles = [f"{m} Comparison" for m in metrics_to_plot]
    if has_cost:
        subplot_titles.append("Cost Comparison")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
    )

    # Add metric comparisons
    for idx, metric in enumerate(metrics_to_plot):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        run_ids = [exp.run_id for exp in comparison.experiments]
        values = [exp.get_metric(metric) or 0.0 for exp in comparison.experiments]

        fig.add_trace(
            go.Bar(
                x=run_ids,
                y=values,
                name=metric,
                text=[f"{v:.4f}" for v in values],
                textposition="auto",
                hovertemplate=(f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"),
            ),
            row=row,
            col=col,
        )

    # Add cost comparison if available
    if has_cost:
        idx = len(metrics_to_plot)
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        run_ids = [exp.run_id for exp in comparison.experiments]
        costs = [exp.get_cost() or 0.0 for exp in comparison.experiments]

        fig.add_trace(
            go.Bar(
                x=run_ids,
                y=costs,
                name="Cost",
                text=[f"${v:.4f}" for v in costs],
                textposition="auto",
                marker_color="green",
                hovertemplate="<b>%{x}</b><br>Cost: $%{y:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="Experiment Dashboard",
        template="plotly_white",
        font=dict(size=12),
        showlegend=False,
        height=400 * rows,
    )

    return fig
