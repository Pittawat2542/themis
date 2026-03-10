import pytest
from themis.report.exporters import CsvExporter, MarkdownExporter, LatexExporter
from themis.records.report import EvaluationReport, ReportTable, ReportMetadata
import pandas as pd
import tempfile


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


def test_csv_exporter(mock_report):
    exporter = CsvExporter()
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        exporter.export(mock_report, f.name)

        # Read back
        df = pd.read_csv(f.name)
        row = df[df["Model"] == "gpt-4"].iloc[0]
        assert row["Score"] == 0.95


def test_markdown_exporter(mock_report):
    exporter = MarkdownExporter()
    with tempfile.NamedTemporaryFile(suffix=".md") as f:
        exporter.export(mock_report, f.name)

        with open(f.name, "r") as r:
            content = r.read()
            assert "# Evaluation Report" in content
            assert "## Main Results" in content
            assert "Core scores." in content
            assert "gpt-4" in content
            assert "0.95" in content
            assert "Environment: Test" in content


def test_latex_exporter(mock_report):
    exporter = LatexExporter()
    with tempfile.NamedTemporaryFile(suffix=".tex") as f:
        exporter.export(mock_report, f.name)

        with open(f.name, "r") as r:
            content = r.read()
            assert "\\section*{Evaluation Report}" in content
            assert "\\subsection*{Main Results}" in content
            assert "gpt-4" in content
            assert "0.95" in content
