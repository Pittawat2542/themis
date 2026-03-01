import pytest
from themis.experiment.comparison import MultiExperimentComparison, ComparisonRow


@pytest.fixture
def sample_comparison():
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
    ]
    return MultiExperimentComparison(
        experiments=experiments, metrics=["accuracy", "f1"]
    )


def test_export_to_csv(sample_comparison, tmp_path):
    output_path = tmp_path / "comparison.csv"
    sample_comparison.to_csv(output_path, include_metadata=True)
    assert output_path.exists()
    content = output_path.read_text()
    assert "run-1" in content
    assert "gpt-4" in content

    # Test without metadata
    output_path2 = tmp_path / "comparison2.csv"
    sample_comparison.to_csv(output_path2, include_metadata=False)
    content2 = output_path2.read_text()
    assert "claude-3-opus" not in content2


def test_export_to_markdown(sample_comparison, tmp_path):
    # Without output path
    md = sample_comparison.to_markdown()
    assert "| Run ID |" in md
    assert "run-1" in md

    # With output path
    output_path = tmp_path / "comparison.md"
    sample_comparison.to_markdown(output_path)
    assert output_path.exists()
    assert "run-2" in output_path.read_text()


def test_export_to_latex(sample_comparison, tmp_path):
    # Basic style
    latex_basic = sample_comparison.to_latex(
        style="basic", caption="Test Cap", label="tab:lbl"
    )
    assert "\\begin{table}" in latex_basic
    assert "Test Cap" in latex_basic
    assert "tab:lbl" in latex_basic
    assert "run-1" in latex_basic

    # Booktabs style
    latex_booktabs = sample_comparison.to_latex(style="booktabs")
    assert "\\toprule" in latex_booktabs

    # With output path
    output_path = tmp_path / "comparison.tex"
    sample_comparison.to_latex(output_path)
    assert output_path.exists()


def test_export_to_dict(sample_comparison):
    data = sample_comparison.to_dict()
    assert data["metrics"] == ["accuracy", "f1"]
    assert len(data["experiments"]) == 2
