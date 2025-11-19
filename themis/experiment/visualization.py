"""Interactive visualizations for experiments using Plotly."""

from __future__ import annotations

from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    px = None  # type: ignore
    make_subplots = None  # type: ignore

from themis.experiment.comparison import MultiExperimentComparison
from themis.experiment.cost import CostBreakdown
from themis.evaluation.reports import EvaluationReport


def _check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


class InteractiveVisualizer:
    """Create interactive visualizations for experiments using Plotly.

    Example:
        >>> visualizer = InteractiveVisualizer()
        >>> fig = visualizer.plot_metric_comparison(comparison, "accuracy")
        >>> fig.write_html("comparison.html")
    """

    def __init__(self):
        """Initialize visualizer."""
        _check_plotly()

    def plot_metric_comparison(
        self,
        comparison: MultiExperimentComparison,
        metric: str,
        title: str | None = None,
        show_values: bool = True,
    ) -> go.Figure:
        """Create bar chart comparing metric across experiments.

        Args:
            comparison: Multi-experiment comparison
            metric: Metric name to visualize
            title: Chart title (default: "{metric} Comparison")
            show_values: Show values on bars

        Returns:
            Plotly Figure object

        Example:
            >>> fig = visualizer.plot_metric_comparison(comparison, "accuracy")
            >>> fig.show()
        """
        if metric not in comparison.metrics and metric not in ("cost", "total_cost"):
            raise ValueError(
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
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<br>"
                    "<extra></extra>",
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

    def plot_pareto_frontier(
        self,
        comparison: MultiExperimentComparison,
        metric1: str,
        metric2: str,
        pareto_ids: list[str],
        maximize1: bool = True,
        maximize2: bool = True,
        title: str | None = None,
    ) -> go.Figure:
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

        Example:
            >>> pareto = comparison.pareto_frontier(["accuracy", "cost"], [True, False])
            >>> fig = visualizer.plot_pareto_frontier(
            ...     comparison, "accuracy", "cost", pareto, True, False
            ... )
        """
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

    def plot_metric_distribution(
        self,
        report: EvaluationReport,
        metric: str,
        plot_type: str = "histogram",
        title: str | None = None,
    ) -> go.Figure:
        """Create histogram or violin plot of metric distribution.

        Args:
            report: Evaluation report
            metric: Metric name
            plot_type: "histogram", "box", or "violin"
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> fig = visualizer.plot_metric_distribution(report, "accuracy")
            >>> fig = visualizer.plot_metric_distribution(report, "accuracy", "violin")
        """
        if metric not in report.metrics:
            raise ValueError(
                f"Metric '{metric}' not found. Available: {list(report.metrics.keys())}"
            )

        # Extract metric values per sample
        values = []
        for record in report.records:
            for score in record.scores:
                if score.metric_name == metric:
                    values.append(score.value)

        if not values:
            raise ValueError(f"No values found for metric '{metric}'")

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
            raise ValueError(
                f"Unknown plot_type '{plot_type}'. Use 'histogram', 'box', or 'violin'"
            )

        fig.update_layout(
            title=title or f"{metric} Distribution ({len(values)} samples)",
            template="plotly_white",
            font=dict(size=12),
        )

        return fig

    def plot_cost_breakdown(
        self,
        cost_breakdown: CostBreakdown,
        title: str | None = None,
    ) -> go.Figure:
        """Create pie chart of cost breakdown.

        Args:
            cost_breakdown: Cost breakdown data
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> breakdown = tracker.get_breakdown()
            >>> fig = visualizer.plot_cost_breakdown(breakdown)
        """
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

    def plot_metric_evolution(
        self,
        comparison: MultiExperimentComparison,
        metric: str,
        title: str | None = None,
    ) -> go.Figure:
        """Create line plot showing metric evolution across runs.

        Experiments are ordered by timestamp if available.

        Args:
            comparison: Multi-experiment comparison
            metric: Metric name
            title: Chart title

        Returns:
            Plotly Figure object

        Example:
            >>> fig = visualizer.plot_metric_evolution(comparison, "accuracy")
        """
        if metric not in comparison.metrics and metric not in ("cost", "total_cost"):
            raise ValueError(
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

    def create_dashboard(
        self,
        comparison: MultiExperimentComparison,
        metrics: list[str] | None = None,
        include_cost: bool = True,
    ) -> go.Figure:
        """Create comprehensive dashboard with multiple charts.

        Args:
            comparison: Multi-experiment comparison
            metrics: Metrics to visualize (default: all)
            include_cost: Include cost visualization if available

        Returns:
            Plotly Figure with subplots

        Example:
            >>> fig = visualizer.create_dashboard(comparison)
            >>> fig.write_html("dashboard.html")
        """
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
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
                    ),
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


def export_interactive_html(
    fig: go.Figure,
    output_path: Path | str,
    include_plotlyjs: str = "cdn",
) -> None:
    """Export Plotly figure to standalone HTML file.

    Args:
        fig: Plotly Figure object
        output_path: Where to save HTML file
        include_plotlyjs: How to include Plotly.js
            - "cdn": Link to CDN (smaller file, requires internet)
            - True: Embed full library (larger file, works offline)
            - False: Don't include (for embedding in existing HTML)

    Example:
        >>> fig = visualizer.plot_metric_comparison(comparison, "accuracy")
        >>> export_interactive_html(fig, "comparison.html")
    """
    _check_plotly()
    output_path = Path(output_path)
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs)


__all__ = [
    "InteractiveVisualizer",
    "export_interactive_html",
    "PLOTLY_AVAILABLE",
]
