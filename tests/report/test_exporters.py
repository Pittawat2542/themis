import pytest
from themis.report.exporters import CsvExporter, MarkdownExporter, LatexExporter
from themis.records.report import EvaluationReport, ReportTable, ReportMetadata
import pandas as pd
from pathlib import Path


@pytest.fixture
def mock_report():
    data = [{"Model": "gpt-4", "Score": 0.95}, {"Model": "gpt-3.5", "Score": 0.85}]

    t1 = ReportTable(
        spec_hash="tbl1",
        id="main",
        title="Main Results",
        description="Core scores.",
        data=data,
    )

    meta = ReportMetadata(
        spec_hash="meta1",
        themis_version="1.5.0",
        spec_hashes=["abcdef"],
        extras={"Environment": "Test"},
    )

    return EvaluationReport(spec_hash="rpt1", tables=[t1], metadata=meta)


def test_csv_exporter(mock_report, tmp_path: Path):
    exporter = CsvExporter()
    output_path = tmp_path / "report.csv"
    exporter.export(mock_report, str(output_path))

    df = pd.read_csv(output_path)
    row = df[df["Model"] == "gpt-4"].iloc[0]
    assert row["Score"] == 0.95


def test_markdown_exporter(mock_report, tmp_path: Path):
    exporter = MarkdownExporter()
    output_path = tmp_path / "report.md"
    exporter.export(mock_report, str(output_path))

    content = output_path.read_text()
    assert "# Evaluation Report" in content
    assert "## Main Results" in content
    assert "Core scores." in content
    assert "gpt-4" in content
    assert "0.95" in content
    assert "Environment: Test" in content


def test_latex_exporter(mock_report, tmp_path: Path):
    exporter = LatexExporter()
    output_path = tmp_path / "report.tex"
    exporter.export(mock_report, str(output_path))

    content = output_path.read_text()
    assert "\\section*{Evaluation Report}" in content
    assert "\\subsection*{Main Results}" in content
    assert "gpt-4" in content
    assert "0.95" in content
