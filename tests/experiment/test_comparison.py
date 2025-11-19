"""Tests for multi-experiment comparison functionality."""

import json
from pathlib import Path

import pytest

from themis.experiment.comparison import (
    ComparisonRow,
    ConfigDiff,
    MultiExperimentComparison,
    compare_experiments,
    diff_configs,
)


@pytest.fixture
def mock_report_dir(tmp_path: Path) -> Path:
    """Create mock experiment reports for testing."""
    # Create three mock experiment runs
    runs = [
        {
            "run_id": "run-1",
            "metrics": [
                {"name": "accuracy", "count": 10, "mean": 0.85},
                {"name": "f1_score", "count": 10, "mean": 0.82},
            ],
            "total_samples": 10,
            "summary": {
                "run_failures": 1,
                "evaluation_failures": 0,
                "model": "gpt-4",
                "temperature": 0.0,
            },
        },
        {
            "run_id": "run-2",
            "metrics": [
                {"name": "accuracy", "count": 10, "mean": 0.90},
                {"name": "f1_score", "count": 10, "mean": 0.88},
            ],
            "total_samples": 10,
            "summary": {
                "run_failures": 0,
                "evaluation_failures": 1,
                "model": "gpt-4",
                "temperature": 0.7,
            },
        },
        {
            "run_id": "run-3",
            "metrics": [
                {"name": "accuracy", "count": 10, "mean": 0.75},
                {"name": "f1_score", "count": 10, "mean": 0.73},
            ],
            "total_samples": 10,
            "summary": {
                "run_failures": 2,
                "evaluation_failures": 1,
                "model": "gpt-3.5-turbo",
                "temperature": 0.0,
            },
        },
    ]

    for run in runs:
        run_dir = tmp_path / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save report.json
        report_path = run_dir / "report.json"
        with report_path.open("w") as f:
            json.dump(run, f, indent=2)

        # Save config.json for diff tests
        config = {
            "model": run["summary"]["model"],
            "temperature": run["summary"]["temperature"],
        }
        config_path = run_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    return tmp_path


def test_comparison_row_get_metric():
    """Test ComparisonRow.get_metric method."""
    row = ComparisonRow(
        run_id="test-run",
        metric_values={"accuracy": 0.85, "f1_score": 0.82},
    )

    assert row.get_metric("accuracy") == 0.85
    assert row.get_metric("f1_score") == 0.82
    assert row.get_metric("missing") is None


def test_multi_experiment_comparison_rank_by_metric():
    """Test ranking experiments by metric value."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85, "cost": 0.05}
        ),
        ComparisonRow(
            run_id="run-2", metric_values={"accuracy": 0.90, "cost": 0.10}
        ),
        ComparisonRow(
            run_id="run-3", metric_values={"accuracy": 0.75, "cost": 0.02}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "cost"]
    )

    # Rank by accuracy (descending)
    ranked = comparison.rank_by_metric("accuracy", ascending=False)
    assert [r.run_id for r in ranked] == ["run-2", "run-1", "run-3"]

    # Rank by cost (ascending)
    ranked = comparison.rank_by_metric("cost", ascending=True)
    assert [r.run_id for r in ranked] == ["run-3", "run-1", "run-2"]


def test_multi_experiment_comparison_highlight_best():
    """Test finding best experiment for a metric."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85, "cost": 0.05}
        ),
        ComparisonRow(
            run_id="run-2", metric_values={"accuracy": 0.90, "cost": 0.10}
        ),
        ComparisonRow(
            run_id="run-3", metric_values={"accuracy": 0.75, "cost": 0.02}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "cost"]
    )

    # Best accuracy (higher is better)
    best = comparison.highlight_best("accuracy", higher_is_better=True)
    assert best is not None
    assert best.run_id == "run-2"

    # Best cost (lower is better)
    best = comparison.highlight_best("cost", higher_is_better=False)
    assert best is not None
    assert best.run_id == "run-3"


def test_multi_experiment_comparison_pareto_frontier():
    """Test Pareto frontier computation."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85, "cost": 0.05}
        ),
        ComparisonRow(
            run_id="run-2", metric_values={"accuracy": 0.90, "cost": 0.10}
        ),
        ComparisonRow(
            run_id="run-3", metric_values={"accuracy": 0.75, "cost": 0.02}
        ),
        ComparisonRow(
            run_id="run-4", metric_values={"accuracy": 0.80, "cost": 0.08}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "cost"]
    )

    # Maximize accuracy, minimize cost
    pareto = comparison.pareto_frontier(["accuracy", "cost"], maximize=[True, False])

    # run-2 has highest accuracy (even though high cost)
    # run-3 has lowest cost (even though low accuracy)
    # run-1 is potentially on frontier (good balance)
    # run-4 is dominated by run-1 (worse accuracy, higher cost)
    assert "run-2" in pareto  # Best accuracy
    assert "run-3" in pareto  # Best cost
    assert "run-4" not in pareto  # Dominated


def test_compare_experiments(mock_report_dir: Path):
    """Test loading and comparing multiple experiments."""
    comparison = compare_experiments(
        run_ids=["run-1", "run-2", "run-3"],
        storage_dir=mock_report_dir,
        metrics=None,  # Load all metrics
        include_metadata=True,
    )

    assert len(comparison.experiments) == 3
    assert "accuracy" in comparison.metrics
    assert "f1_score" in comparison.metrics

    # Check metrics loaded correctly
    run1 = next(e for e in comparison.experiments if e.run_id == "run-1")
    assert run1.get_metric("accuracy") == 0.85
    assert run1.sample_count == 10
    assert run1.failure_count == 1


def test_compare_experiments_specific_metrics(mock_report_dir: Path):
    """Test comparing experiments with specific metrics."""
    comparison = compare_experiments(
        run_ids=["run-1", "run-2"],
        storage_dir=mock_report_dir,
        metrics=["accuracy"],  # Only load accuracy
        include_metadata=False,
    )

    assert len(comparison.experiments) == 2
    assert comparison.metrics == ["accuracy"]


def test_compare_experiments_missing_run(mock_report_dir: Path):
    """Test handling missing run IDs."""
    with pytest.raises(ValueError, match="No valid experiments found"):
        compare_experiments(
            run_ids=["missing-run"],
            storage_dir=mock_report_dir,
        )


def test_config_diff(mock_report_dir: Path):
    """Test configuration diffing."""
    diff = diff_configs("run-1", "run-2", mock_report_dir)

    assert diff.run_id_a == "run-1"
    assert diff.run_id_b == "run-2"
    assert diff.has_differences()

    # Temperature changed from 0.0 to 0.7
    assert "temperature" in diff.changed_fields
    assert diff.changed_fields["temperature"] == (0.0, 0.7)

    # Model stayed the same
    assert "model" not in diff.changed_fields


def test_latex_export_booktabs():
    """Test LaTeX export with booktabs style."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85, "f1": 0.82}
        ),
        ComparisonRow(
            run_id="run-2", metric_values={"accuracy": 0.90, "f1": 0.88}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "f1"]
    )

    latex = comparison.to_latex(
        style="booktabs",
        caption="Test results",
        label="tab:test"
    )

    assert "\\begin{table}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\caption{Test results}" in latex
    assert "\\label{tab:test}" in latex
    assert "0.8500" in latex  # accuracy value
    assert "0.9000" in latex  # accuracy value


def test_latex_export_basic():
    """Test LaTeX export with basic style."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    latex = comparison.to_latex(style="basic")

    assert "\\begin{table}" in latex
    assert "\\hline" in latex
    assert "0.8500" in latex


def test_latex_export_with_cost():
    """Test LaTeX export includes cost data."""
    experiments = [
        ComparisonRow(
            run_id="run-1",
            metric_values={"accuracy": 0.85},
            metadata={"cost": {"total_cost": 0.05}}
        ),
        ComparisonRow(
            run_id="run-2",
            metric_values={"accuracy": 0.90},
            metadata={"cost": {"total_cost": 0.10}}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    latex = comparison.to_latex()

    assert "Cost (\\$)" in latex
    assert "0.0500" in latex  # cost value
    assert "0.1000" in latex  # cost value


def test_latex_export_escapes_underscores():
    """Test that underscores in run IDs are escaped."""
    experiments = [
        ComparisonRow(
            run_id="my_run_id", metric_values={"accuracy": 0.85}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    latex = comparison.to_latex()

    assert "my\\_run\\_id" in latex
    assert "my_run_id" not in latex.replace("\\_", "X")  # No unescaped underscores


def test_latex_export_to_file(tmp_path):
    """Test LaTeX export to file."""
    experiments = [
        ComparisonRow(
            run_id="run-1", metric_values={"accuracy": 0.85}
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    output_path = tmp_path / "table.tex"
    latex = comparison.to_latex(output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert content == latex
    assert "\\begin{table}" in content


def test_config_diff_identical(mock_report_dir: Path):
    """Test diffing identical configurations."""
    diff = diff_configs("run-1", "run-1", mock_report_dir)

    assert not diff.has_differences()
    assert len(diff.changed_fields) == 0
    assert len(diff.added_fields) == 0
    assert len(diff.removed_fields) == 0


def test_comparison_to_csv(tmp_path: Path):
    """Test exporting comparison to CSV."""
    experiments = [
        ComparisonRow(
            run_id="run-1",
            metric_values={"accuracy": 0.85},
            sample_count=10,
            failure_count=1,
        ),
        ComparisonRow(
            run_id="run-2",
            metric_values={"accuracy": 0.90},
            sample_count=10,
            failure_count=0,
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    csv_path = tmp_path / "comparison.csv"
    comparison.to_csv(csv_path, include_metadata=True)

    assert csv_path.exists()

    # Verify CSV content
    import csv

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert float(rows[0]["accuracy"]) == 0.85
    assert rows[1]["run_id"] == "run-2"
    assert float(rows[1]["accuracy"]) == 0.90


def test_comparison_to_markdown(tmp_path: Path):
    """Test exporting comparison to Markdown."""
    experiments = [
        ComparisonRow(
            run_id="run-1",
            metric_values={"accuracy": 0.85},
            sample_count=10,
            failure_count=1,
        ),
        ComparisonRow(
            run_id="run-2",
            metric_values={"accuracy": 0.90},
            sample_count=10,
            failure_count=0,
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    md_path = tmp_path / "comparison.md"
    markdown = comparison.to_markdown(md_path)

    assert md_path.exists()
    assert "# Experiment Comparison" in markdown
    assert "run-1" in markdown
    assert "run-2" in markdown
    assert "0.8500" in markdown
    assert "0.9000" in markdown


def test_comparison_to_dict():
    """Test exporting comparison to dictionary."""
    experiments = [
        ComparisonRow(
            run_id="run-1",
            metric_values={"accuracy": 0.85},
            sample_count=10,
            failure_count=1,
        ),
    ]

    comparison = MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy"]
    )

    data = comparison.to_dict()

    assert "experiments" in data
    assert "metrics" in data
    assert len(data["experiments"]) == 1
    assert data["experiments"][0]["run_id"] == "run-1"
    assert data["metrics"] == ["accuracy"]
