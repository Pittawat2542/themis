"""Pareto frontier visualizations."""

from __future__ import annotations

from themis.experiment.comparison import MultiExperimentComparison
from themis.experiment.visualization.core import _check_plotly, go


def plot_pareto_frontier(
    comparison: MultiExperimentComparison,
    metric1: str,
    metric2: str,
    pareto_ids: list[str],
    maximize1: bool = True,
    maximize2: bool = True,
    title: str | None = None,
) -> "go.Figure":
    """Create scatter plot with Pareto frontier highlighted.

    Args:
        comparison: Multi-experiment comparison
        metric1: First metric (x-axis)
        metric2: Second metric (y-axis)
        pareto_ids: Run IDs on Pareto frontier
        maximize1: Whether metric1 should be maximized
        maximize2: Whether metric2 should be maximized
        title: Chart title

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    # Extract data
    x_values = []
    y_values = []
    run_ids = []
    is_pareto = []

    for exp in comparison.experiments:
        x_val = exp.get_metric(metric1)
        y_val = exp.get_metric(metric2)

        if x_val is not None and y_val is not None:
            x_values.append(x_val)
            y_values.append(y_val)
            run_ids.append(exp.run_id)
            is_pareto.append(exp.run_id in pareto_ids)

    # Create scatter plot
    colors = ["red" if p else "blue" for p in is_pareto]
    sizes = [12 if p else 8 for p in is_pareto]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=run_ids,
                textposition="top center",
                marker=dict(
                    color=colors,
                    size=sizes,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>%{text}</b><br>"
                + f"{metric1}: %{{x:.4f}}<br>"
                + f"{metric2}: %{{y:.4f}}<br>"
                + "<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title or f"Pareto Frontier: {metric1} vs {metric2}",
        xaxis_title=f"{metric1} ({'maximize' if maximize1 else 'minimize'})",
        yaxis_title=f"{metric2} ({'maximize' if maximize2 else 'minimize'})",
        template="plotly_white",
        font=dict(size=12),
        showlegend=False,
    )

    # Add legend for colors
    fig.add_annotation(
        text="<b style='color:red'>●</b> Pareto optimal<br>"
        "<b style='color:blue'>●</b> Dominated",
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    return fig
