"""Cost breakdown visualizations."""

from __future__ import annotations

from themis.experiment.cost import CostBreakdown
from themis.experiment.visualization.core import _check_plotly, go, make_subplots


def plot_cost_breakdown(
    cost_breakdown: CostBreakdown,
    title: str | None = None,
) -> "go.Figure":
    """Create pie chart of cost breakdown.

    Args:
        cost_breakdown: Cost breakdown data
        title: Chart title

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    # Build data for pie chart
    labels = []
    values = []

    # Generation vs Evaluation
    if cost_breakdown.generation_cost > 0:
        labels.append("Generation")
        values.append(cost_breakdown.generation_cost)

    if cost_breakdown.evaluation_cost > 0:
        labels.append("Evaluation")
        values.append(cost_breakdown.evaluation_cost)

    # If we have per-model breakdown, create a second pie
    if cost_breakdown.per_model_costs:
        # Create subplots for overall and per-model
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Cost by Phase", "Cost by Model"),
            specs=[[{"type": "pie"}, {"type": "pie"}]],
        )

        # Overall breakdown
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b><br>"
                "Cost: $%{value:.4f}<br>"
                "Percentage: %{percent}<br>"
                "<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Per-model breakdown
        model_labels = list(cost_breakdown.per_model_costs.keys())
        model_values = list(cost_breakdown.per_model_costs.values())

        fig.add_trace(
            go.Pie(
                labels=model_labels,
                values=model_values,
                textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b><br>"
                "Cost: $%{value:.4f}<br>"
                "Percentage: %{percent}<br>"
                "<extra></extra>",
            ),
            row=1,
            col=2,
        )

        default_title = f"Cost Breakdown (Total: ${cost_breakdown.total_cost:.4f})"
        fig.update_layout(
            title_text=title or default_title,
            template="plotly_white",
            font=dict(size=12),
        )
    else:
        # Single pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>"
                    "Cost: $%{value:.4f}<br>"
                    "Percentage: %{percent}<br>"
                    "<extra></extra>",
                )
            ]
        )

        default_title = f"Cost Breakdown (Total: ${cost_breakdown.total_cost:.4f})"
        fig.update_layout(
            title=title or default_title,
            template="plotly_white",
            font=dict(size=12),
        )

    return fig
