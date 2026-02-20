"""Tests for interactive visualizations."""

import pytest

from themis.experiment.comparison import ComparisonRow, MultiExperimentComparison
from themis.experiment.cost import CostTracker
from themis.experiment.visualization import PLOTLY_AVAILABLE

# Skip all tests if plotly is not available
pytestmark = pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")


@pytest.fixture
def sample_comparison():
    """Create sample comparison for testing."""
    experiments = [
        ComparisonRow(
            run_id="run-1",
            metric_values={"accuracy": 0.85, "f1": 0.82},
            metadata={"model": "gpt-4"},
            sample_count=100,
        ),
        ComparisonRow(
            run_id="run-2",
            metric_values={"accuracy": 0.90, "f1": 0.88},
            metadata={"model": "claude-3-opus", "cost": {"total_cost": 0.05}},
            sample_count=100,
        ),
        ComparisonRow(
            run_id="run-3",
            metric_values={"accuracy": 0.75, "f1": 0.72},
            metadata={"model": "gpt-3.5-turbo", "cost": {"total_cost": 0.02}},
            sample_count=100,
        ),
    ]
    return MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "f1"]
    )


@pytest.fixture
def sample_cost_breakdown():
    """Create sample cost breakdown."""
    tracker = CostTracker()
    tracker.record_generation("gpt-4", 1000, 500, 0.06)
    tracker.record_generation("gpt-4", 800, 400, 0.048)
    tracker.record_generation("gpt-3.5-turbo", 1000, 500, 0.00125)
    return tracker.get_breakdown()


def test_plotly_available():
    """Test that plotly is available in test environment."""
    assert PLOTLY_AVAILABLE, "Plotly should be available for visualization tests"


def test_visualizer_import():
    """Test that visualizer can be imported."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    assert visualizer is not None


def test_plot_metric_comparison(sample_comparison):
    """Test metric comparison bar chart."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_comparison(sample_comparison, "accuracy")

    assert fig is not None
    assert fig.layout.title.text == "accuracy Comparison"
    assert len(fig.data) == 1  # One bar chart
    assert len(fig.data[0].x) == 3  # Three experiments


def test_plot_metric_comparison_with_cost(sample_comparison):
    """Test metric comparison with cost metric."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_comparison(sample_comparison, "cost")

    assert fig is not None
    assert len(fig.data[0].x) == 3


def test_plot_metric_comparison_invalid_metric(sample_comparison):
    """Test error handling for invalid metric."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()

    with pytest.raises(ValueError, match="not found"):
        visualizer.plot_metric_comparison(sample_comparison, "invalid_metric")


def test_plot_pareto_frontier(sample_comparison):
    """Test Pareto frontier visualization."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()

    # Compute Pareto frontier
    pareto_ids = sample_comparison.pareto_frontier(["accuracy", "cost"], [True, False])

    # Create visualization
    fig = visualizer.plot_pareto_frontier(
        sample_comparison, "accuracy", "cost", pareto_ids, True, False
    )

    assert fig is not None
    assert "Pareto Frontier" in fig.layout.title.text
    assert len(fig.data) == 1  # One scatter plot


def test_plot_metric_evolution(sample_comparison):
    """Test metric evolution line chart."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_evolution(sample_comparison, "accuracy")

    assert fig is not None
    assert "Evolution" in fig.layout.title.text
    assert len(fig.data) == 1  # One line chart
    assert len(fig.data[0].x) == 3  # Three experiments


def test_plot_cost_breakdown(sample_cost_breakdown):
    """Test cost breakdown pie chart."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_cost_breakdown(sample_cost_breakdown)

    assert fig is not None
    assert "Cost Breakdown" in fig.layout.title.text


def test_plot_cost_breakdown_with_models():
    """Test cost breakdown with per-model data."""
    from themis.experiment.visualization import InteractiveVisualizer

    tracker = CostTracker()
    tracker.record_generation("gpt-4", 1000, 500, 0.06)
    tracker.record_generation("gpt-3.5-turbo", 1000, 500, 0.00125)

    breakdown = tracker.get_breakdown()

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_cost_breakdown(breakdown)

    assert fig is not None
    # Should have two pie charts (overall and per-model)
    assert len(fig.data) == 2


def test_create_dashboard(sample_comparison):
    """Test dashboard creation."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.create_dashboard(sample_comparison)

    assert fig is not None
    assert "Dashboard" in fig.layout.title.text


def test_create_dashboard_with_cost(sample_comparison):
    """Test dashboard with cost data."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.create_dashboard(sample_comparison, include_cost=True)

    assert fig is not None


def test_export_interactive_html(tmp_path, sample_comparison):
    """Test HTML export."""
    from themis.experiment.visualization import (
        InteractiveVisualizer,
        export_interactive_html,
    )

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_comparison(sample_comparison, "accuracy")

    output_path = tmp_path / "test.html"
    export_interactive_html(fig, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "plotly" in content.lower()
    assert "accuracy" in content.lower()


def test_plot_metric_comparison_custom_title(sample_comparison):
    """Test custom title for metric comparison."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_comparison(
        sample_comparison, "accuracy", title="Custom Title"
    )

    assert fig.layout.title.text == "Custom Title"


def test_plot_metric_comparison_no_values(sample_comparison):
    """Test metric comparison without showing values."""
    from themis.experiment.visualization import InteractiveVisualizer

    visualizer = InteractiveVisualizer()
    fig = visualizer.plot_metric_comparison(
        sample_comparison, "accuracy", show_values=False
    )

    assert fig is not None
    assert fig.data[0].text is None


def test_visualizer_without_plotly():
    """Test error handling when plotly is not available."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"plotly": None}):
        # Reimport with plotly unavailable
        from importlib import reload
        from themis.experiment import visualization

        reload(visualization)

        # Should raise ImportError when creating visualizer
        with pytest.raises(ImportError, match="Plotly is required"):
            visualization.InteractiveVisualizer()
