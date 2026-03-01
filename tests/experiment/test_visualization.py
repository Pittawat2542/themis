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
    from themis.experiment.visualization.metrics import plot_metric_comparison

    fig = plot_metric_comparison(sample_comparison, "accuracy")

    assert fig is not None
    assert fig.layout.title.text == "accuracy Comparison"
    assert len(fig.data) == 1  # One bar chart
    assert len(fig.data[0].x) == 3  # Three experiments


def test_plot_metric_comparison_with_cost(sample_comparison):
    """Test metric comparison with cost metric."""
    from themis.experiment.visualization.metrics import plot_metric_comparison

    fig = plot_metric_comparison(sample_comparison, "cost")

    assert fig is not None
    assert len(fig.data[0].x) == 3


def test_plot_metric_comparison_invalid_metric(sample_comparison):
    """Test error handling for invalid metric."""
    from themis.experiment.visualization.metrics import plot_metric_comparison

    with pytest.raises(ValueError, match="not found"):
        plot_metric_comparison(sample_comparison, "invalid_metric")


def test_plot_pareto_frontier(sample_comparison):
    """Test Pareto frontier visualization."""
    from themis.experiment.visualization.pareto import plot_pareto_frontier

    # Compute Pareto frontier
    pareto_ids = sample_comparison.pareto_frontier(["accuracy", "cost"], [True, False])

    # Create visualization
    fig = plot_pareto_frontier(
        sample_comparison, "accuracy", "cost", pareto_ids, True, False
    )

    assert fig is not None
    assert "Pareto Frontier" in fig.layout.title.text
    assert len(fig.data) == 1  # One scatter plot


def test_plot_metric_evolution(sample_comparison):
    """Test metric evolution line chart."""
    from themis.experiment.visualization.metrics import plot_metric_evolution

    fig = plot_metric_evolution(sample_comparison, "accuracy")

    assert fig is not None
    assert "Evolution" in fig.layout.title.text
    assert len(fig.data) == 1  # One line chart
    assert len(fig.data[0].x) == 3  # Three experiments


def test_plot_cost_breakdown(sample_cost_breakdown):
    """Test cost breakdown pie chart."""
    from themis.experiment.visualization.cost import plot_cost_breakdown

    fig = plot_cost_breakdown(sample_cost_breakdown)

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
    from themis.experiment.visualization.dashboard import create_dashboard

    fig = create_dashboard(sample_comparison)

    assert fig is not None
    assert "Dashboard" in fig.layout.title.text


def test_create_dashboard_with_cost(sample_comparison):
    """Test dashboard with cost data."""
    from themis.experiment.visualization.dashboard import create_dashboard

    fig = create_dashboard(sample_comparison, include_cost=True)

    assert fig is not None


def test_export_interactive_html(tmp_path, sample_comparison):
    """Test HTML export."""
    from themis.experiment.visualization.html import export_interactive_html
    from themis.experiment.visualization.metrics import plot_metric_comparison

    fig = plot_metric_comparison(sample_comparison, "accuracy")

    output_path = tmp_path / "test.html"
    export_interactive_html(fig, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "plotly" in content.lower()
    assert "accuracy" in content.lower()


def test_plot_metric_comparison_custom_title(sample_comparison):
    """Test custom title for metric comparison."""
    from themis.experiment.visualization.metrics import plot_metric_comparison

    fig = plot_metric_comparison(sample_comparison, "accuracy", title="Custom Title")

    assert fig.layout.title.text == "Custom Title"


def test_plot_metric_comparison_no_values(sample_comparison):
    """Test metric comparison without showing values."""
    from themis.experiment.visualization.metrics import plot_metric_comparison

    fig = plot_metric_comparison(sample_comparison, "accuracy", show_values=False)

    assert fig is not None
    assert fig.data[0].text is None


def test_visualizer_without_plotly():
    """Test error handling when plotly is not available."""
    import sys
    from unittest.mock import patch

    with patch.dict(
        sys.modules,
        {
            "plotly": None,
            "plotly.graph_objects": None,
            "plotly.express": None,
            "plotly.subplots": None,
        },
    ):
        # Reimport with plotly unavailable
        from importlib import reload
        from themis.experiment import visualization
        from themis.experiment.visualization import core

        reload(core)
        reload(visualization)

        # Should raise DependencyError when creating visualizer
        from themis.exceptions import DependencyError

        with pytest.raises(DependencyError, match="Plotly is required"):
            visualization.InteractiveVisualizer()

    # Restore the module state for subsequent tests
    from importlib import reload
    from themis.experiment.visualization import core

    reload(core)
    reload(visualization)


@pytest.fixture
def sample_evaluation_report():
    from themis.core.entities import EvaluationRecord, MetricScore
    from themis.evaluation.reports import EvaluationReport
    from unittest.mock import MagicMock

    records = []

    for i in range(3):
        score = MetricScore(metric_name="accuracy", value=0.5 * (i + 1))
        record = EvaluationRecord(sample_id=str(i), scores=[score], failures=[])
        records.append(record)

    return EvaluationReport(
        metrics={"accuracy": MagicMock()}, records=records, failures=[]
    )


def test_plot_metric_distribution_histogram(sample_evaluation_report):
    from themis.experiment.visualization.metrics import plot_metric_distribution

    fig = plot_metric_distribution(
        sample_evaluation_report, "accuracy", plot_type="histogram"
    )
    assert fig is not None
    assert "accuracy Distribution" in fig.layout.title.text


def test_plot_metric_distribution_box(sample_evaluation_report):
    from themis.experiment.visualization.metrics import plot_metric_distribution

    fig = plot_metric_distribution(
        sample_evaluation_report, "accuracy", plot_type="box"
    )
    assert fig is not None


def test_plot_metric_distribution_violin(sample_evaluation_report):
    from themis.experiment.visualization.metrics import plot_metric_distribution

    fig = plot_metric_distribution(
        sample_evaluation_report, "accuracy", plot_type="violin"
    )
    assert fig is not None


def test_plot_metric_distribution_invalid_type(sample_evaluation_report):
    from themis.experiment.visualization.metrics import plot_metric_distribution

    with pytest.raises(Exception, match="Unknown plot_type"):
        plot_metric_distribution(
            sample_evaluation_report, "accuracy", plot_type="magic"
        )


def test_plot_metric_distribution_missing_metric(sample_evaluation_report):
    from themis.experiment.visualization.metrics import plot_metric_distribution
    from themis.exceptions import MetricError

    with pytest.raises(MetricError):
        plot_metric_distribution(sample_evaluation_report, "fake_metric")


def test_plot_metric_distribution_no_values():
    from themis.experiment.visualization.metrics import plot_metric_distribution
    from themis.evaluation.reports import EvaluationReport
    from unittest.mock import MagicMock
    from themis.exceptions import MetricError

    report = EvaluationReport(
        metrics={"accuracy": MagicMock()}, records=[], failures=[]
    )
    with pytest.raises(MetricError, match="No values found"):
        plot_metric_distribution(report, "accuracy")


def test_plot_cost_breakdown_single_pie():
    from themis.experiment.visualization.cost import plot_cost_breakdown
    from themis.experiment.cost import CostBreakdown

    breakdown = CostBreakdown(
        total_cost=1.0, generation_cost=0.5, evaluation_cost=0.5, per_model_costs={}
    )
    fig = plot_cost_breakdown(breakdown)
    assert fig is not None


def test_plot_metric_evolution_missing_timestamp(sample_comparison):
    from themis.experiment.visualization.metrics import plot_metric_evolution

    # Modify one to have no timestamp
    sample_comparison.experiments[0].timestamp = None
    fig = plot_metric_evolution(sample_comparison, "accuracy")
    assert fig is not None


def test_plot_metric_evolution_missing_metric(sample_comparison):
    from themis.experiment.visualization.metrics import plot_metric_evolution
    from themis.exceptions import MetricError

    with pytest.raises(MetricError, match="not found"):
        plot_metric_evolution(sample_comparison, "invalid_metric")
