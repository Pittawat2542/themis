"""Core utilities for visualization."""

from __future__ import annotations

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


def _check_plotly() -> None:
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        from themis.exceptions import DependencyError

        raise DependencyError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


__all__ = ["PLOTLY_AVAILABLE", "_check_plotly", "go", "px", "make_subplots"]
