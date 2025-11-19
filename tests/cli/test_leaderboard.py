"""Tests for leaderboard generation."""

import pytest

from themis.experiment.comparison import ComparisonRow, MultiExperimentComparison


@pytest.fixture
def sample_ranked_experiments():
    """Create sample experiments for leaderboard testing."""
    return [
        ComparisonRow(
            run_id="run-best",
            metric_values={"accuracy": 0.95, "f1": 0.93},
            metadata={"model": "gpt-4", "temperature": 0.0, "cost": {"total_cost": 0.10}},
            sample_count=100,
            failure_count=0,
        ),
        ComparisonRow(
            run_id="run-medium",
            metric_values={"accuracy": 0.85, "f1": 0.82},
            metadata={"model": "claude-3", "temperature": 0.5, "cost": {"total_cost": 0.05}},
            sample_count=100,
            failure_count=2,
        ),
        ComparisonRow(
            run_id="run-lowest",
            metric_values={"accuracy": 0.75, "f1": 0.72},
            metadata={"model": "gpt-3.5", "temperature": 0.7, "cost": {"total_cost": 0.02}},
            sample_count=100,
            failure_count=5,
        ),
    ]


def test_markdown_leaderboard_basic(sample_ranked_experiments):
    """Test basic markdown leaderboard generation."""
    from themis.cli.commands.leaderboard import _generate_markdown_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    markdown = _generate_markdown_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        title="Test Leaderboard",
        include_cost=True,
        include_metadata=None,
        comparison=comparison,
    )

    assert "# Test Leaderboard" in markdown
    assert "run-best" in markdown
    assert "0.9500" in markdown  # Best accuracy in bold
    assert "| Rank |" in markdown
    assert "| 1 |" in markdown
    assert "| 2 |" in markdown
    assert "| 3 |" in markdown


def test_markdown_leaderboard_with_metadata(sample_ranked_experiments):
    """Test markdown leaderboard with metadata columns."""
    from themis.cli.commands.leaderboard import _generate_markdown_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    markdown = _generate_markdown_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        title="Test Leaderboard",
        include_cost=True,
        include_metadata=["model", "temperature"],
        comparison=comparison,
    )

    assert "| model |" in markdown
    assert "| temperature |" in markdown
    assert "gpt-4" in markdown
    assert "0.0" in markdown


def test_markdown_leaderboard_includes_cost(sample_ranked_experiments):
    """Test that cost column is included when requested."""
    from themis.cli.commands.leaderboard import _generate_markdown_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    markdown = _generate_markdown_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        title="Test Leaderboard",
        include_cost=True,
        include_metadata=None,
        comparison=comparison,
    )

    assert "Cost ($)" in markdown
    assert "0.1000" in markdown  # run-best cost
    assert "0.0500" in markdown  # run-medium cost


def test_latex_leaderboard_basic(sample_ranked_experiments):
    """Test basic LaTeX leaderboard generation."""
    from themis.cli.commands.leaderboard import _generate_latex_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    latex = _generate_latex_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        title="Test Leaderboard",
        include_cost=True,
        include_metadata=None,
        comparison=comparison,
    )

    assert "\\begin{table}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\caption{Test Leaderboard}" in latex
    assert "\\label{tab:leaderboard}" in latex
    assert "\\textbf{0.9500}" in latex  # Best accuracy in bold


def test_latex_leaderboard_escapes_underscores():
    """Test that underscores in run IDs are escaped in LaTeX."""
    from themis.cli.commands.leaderboard import _generate_latex_leaderboard

    # Create experiments with underscores in run IDs
    experiments = [
        ComparisonRow(
            run_id="run_best",  # Has underscore
            metric_values={"accuracy": 0.95},
            sample_count=100,
        ),
        ComparisonRow(
            run_id="run_medium",  # Has underscore
            metric_values={"accuracy": 0.85},
            sample_count=100,
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments,
        metrics=["accuracy"]
    )

    latex = _generate_latex_leaderboard(
        ranked=experiments,
        metric="accuracy",
        title="Test",
        include_cost=False,
        include_metadata=None,
        comparison=comparison,
    )

    assert "run\\_best" in latex
    assert "run\\_medium" in latex
    # Check no unescaped underscores in content (excluding \\ sequences)
    content_without_escapes = latex.replace("\\_", "X")
    assert "run_best" not in content_without_escapes


def test_csv_leaderboard_basic(sample_ranked_experiments):
    """Test CSV leaderboard generation."""
    from themis.cli.commands.leaderboard import _generate_csv_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    csv = _generate_csv_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        include_cost=True,
        include_metadata=None,
        comparison=comparison,
    )

    assert "rank,run_id,accuracy" in csv
    assert "1,run-best,0.95" in csv
    assert "2,run-medium,0.85" in csv
    assert "3,run-lowest,0.75" in csv
    assert "cost" in csv


def test_csv_leaderboard_with_metadata(sample_ranked_experiments):
    """Test CSV leaderboard includes metadata."""
    from themis.cli.commands.leaderboard import _generate_csv_leaderboard

    comparison = MultiExperimentComparison(
        experiments=sample_ranked_experiments,
        metrics=["accuracy", "f1"]
    )

    csv = _generate_csv_leaderboard(
        ranked=sample_ranked_experiments,
        metric="accuracy",
        include_cost=True,
        include_metadata=["model", "temperature"],
        comparison=comparison,
    )

    assert "model" in csv
    assert "temperature" in csv
    assert "gpt-4" in csv
    assert "0.0" in csv


def test_leaderboard_ranking_order():
    """Test that leaderboard ranks correctly."""
    experiments = [
        ComparisonRow(run_id="run-3", metric_values={"accuracy": 0.70}),
        ComparisonRow(run_id="run-1", metric_values={"accuracy": 0.90}),
        ComparisonRow(run_id="run-2", metric_values={"accuracy": 0.80}),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments,
        metrics=["accuracy"]
    )

    # Rank descending (default)
    ranked = comparison.rank_by_metric("accuracy", ascending=False)

    assert ranked[0].run_id == "run-1"  # Rank 1: 0.90
    assert ranked[1].run_id == "run-2"  # Rank 2: 0.80
    assert ranked[2].run_id == "run-3"  # Rank 3: 0.70


def test_leaderboard_ascending_ranking():
    """Test ascending ranking (for cost, lower is better)."""
    experiments = [
        ComparisonRow(
            run_id="run-expensive",
            metric_values={},
            metadata={"cost": {"total_cost": 0.10}}
        ),
        ComparisonRow(
            run_id="run-cheap",
            metric_values={},
            metadata={"cost": {"total_cost": 0.02}}
        ),
        ComparisonRow(
            run_id="run-medium",
            metric_values={},
            metadata={"cost": {"total_cost": 0.05}}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments,
        metrics=[]
    )

    # Rank by cost ascending (lower is better)
    ranked = comparison.rank_by_metric("cost", ascending=True)

    assert ranked[0].run_id == "run-cheap"      # Rank 1: 0.02
    assert ranked[1].run_id == "run-medium"     # Rank 2: 0.05
    assert ranked[2].run_id == "run-expensive"  # Rank 3: 0.10


def test_markdown_leaderboard_no_cost_when_unavailable():
    """Test that cost column is omitted when no experiments have cost data."""
    from themis.cli.commands.leaderboard import _generate_markdown_leaderboard

    experiments = [
        ComparisonRow(run_id="run-1", metric_values={"accuracy": 0.90}),
        ComparisonRow(run_id="run-2", metric_values={"accuracy": 0.80}),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments,
        metrics=["accuracy"]
    )

    markdown = _generate_markdown_leaderboard(
        ranked=experiments,
        metric="accuracy",
        title="Test",
        include_cost=True,  # Requested but unavailable
        include_metadata=None,
        comparison=comparison,
    )

    # Cost column should NOT be present
    assert "Cost ($)" not in markdown
