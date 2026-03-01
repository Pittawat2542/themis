"""Interactive visualizations for experiments using Plotly (Facade module).

This module has been refactored. The core visualization logic now lives in the
`themis.experiment.visualization` subpackage.

This file serves as a facade to maintain backward compatibility for existing code
using the monolithic `InteractiveVisualizer` class.
"""

from __future__ import annotations

from themis.evaluation.reports import EvaluationReport
from themis.experiment.comparison import MultiExperimentComparison
from themis.experiment.cost import CostBreakdown
from themis.experiment.visualization.core import PLOTLY_AVAILABLE, _check_plotly
from themis.experiment.visualization.cost import plot_cost_breakdown
from themis.experiment.visualization.dashboard import create_dashboard
from themis.experiment.visualization.html import export_interactive_html
from themis.experiment.visualization.metrics import (
    plot_metric_comparison,
    plot_metric_distribution,
    plot_metric_evolution,
)
from themis.experiment.visualization.pareto import plot_pareto_frontier

if PLOTLY_AVAILABLE:
    from themis.experiment.visualization.core import go


class InteractiveVisualizer:
    """Create interactive visualizations for experiments using Plotly.

    Note: This class is preserved for backward compatibility. It perfectly delegates
    to the modular functional API in `themis.experiment.visualization.*`.
    """

    def __init__(self) -> None:
        """Initialize visualizer."""
        _check_plotly()

    def plot_metric_comparison(
        self,
        comparison: MultiExperimentComparison,
        metric: str,
        title: str | None = None,
        show_values: bool = True,
    ) -> "go.Figure":
        return plot_metric_comparison(
            comparison=comparison,
            metric=metric,
            title=title,
            show_values=show_values,
        )

    def plot_pareto_frontier(
        self,
        comparison: MultiExperimentComparison,
        metric1: str,
        metric2: str,
        pareto_ids: list[str],
        maximize1: bool = True,
        maximize2: bool = True,
        title: str | None = None,
    ) -> "go.Figure":
        return plot_pareto_frontier(
            comparison=comparison,
            metric1=metric1,
            metric2=metric2,
            pareto_ids=pareto_ids,
            maximize1=maximize1,
            maximize2=maximize2,
            title=title,
        )

    def plot_metric_distribution(
        self,
        report: EvaluationReport,
        metric: str,
        plot_type: str = "histogram",
        title: str | None = None,
    ) -> "go.Figure":
        return plot_metric_distribution(
            report=report,
            metric=metric,
            plot_type=plot_type,
            title=title,
        )

    def plot_cost_breakdown(
        self,
        cost_breakdown: CostBreakdown,
        title: str | None = None,
    ) -> "go.Figure":
        return plot_cost_breakdown(
            cost_breakdown=cost_breakdown,
            title=title,
        )

    def plot_metric_evolution(
        self,
        comparison: MultiExperimentComparison,
        metric: str,
        title: str | None = None,
    ) -> "go.Figure":
        return plot_metric_evolution(
            comparison=comparison,
            metric=metric,
            title=title,
        )

    def create_dashboard(
        self,
        comparison: MultiExperimentComparison,
        metrics: list[str] | None = None,
        include_cost: bool = True,
    ) -> "go.Figure":
        return create_dashboard(
            comparison=comparison,
            metrics=metrics,
            include_cost=include_cost,
        )


__all__ = [
    "InteractiveVisualizer",
    "export_interactive_html",
    "PLOTLY_AVAILABLE",
]
