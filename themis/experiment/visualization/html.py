"""HTML export utilities for visualizations."""

from __future__ import annotations

from pathlib import Path

from themis.experiment.visualization.core import _check_plotly, go


def export_interactive_html(
    fig: "go.Figure",
    output_path: Path | str,
    include_plotlyjs: str | bool = "cdn",
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
        >>> fig = plot_metric_comparison(comparison, "accuracy")
        >>> export_interactive_html(fig, "comparison.html")
    """
    _check_plotly()
    output_path = Path(output_path)
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs)
