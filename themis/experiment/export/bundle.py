"""Bundle export utilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

from themis.experiment import orchestrator
from themis.experiment.export._shared import ChartLike
from themis.experiment.export.csv import export_report_csv
from themis.experiment.export.html import export_html_report
from themis.experiment.export.json import export_report_json, export_summary_json


def export_report_bundle(
    report: orchestrator.ExperimentReport,
    *,
    csv_path: str | Path | None = None,
    html_path: str | Path | None = None,
    json_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    run_id: str | None = None,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int = 100,
    indent: int = 2,
) -> OrderedDict[str, Path]:
    """Convenience helper that writes multiple export formats at once.

    Args:
        report: Experiment report to export
        csv_path: Optional path for CSV export
        html_path: Optional path for HTML export
        json_path: Optional path for full JSON export
        summary_path: Optional path for lightweight summary JSON export
        run_id: Optional run identifier for summary
        charts: Optional charts to include in visualizations
        title: Report title
        sample_limit: Maximum samples to include in detailed exports
        indent: JSON indentation level

    Returns:
        Ordered dict of format -> path for created files

    Note:
        The summary export is highly recommended as it provides quick access
        to key metrics without parsing large report files.
    """
    outputs: OrderedDict[str, Path] = OrderedDict()
    if csv_path is not None:
        outputs["csv"] = export_report_csv(report, csv_path)
    if html_path is not None:
        outputs["html"] = export_html_report(
            report,
            html_path,
            charts=charts,
            title=title,
            sample_limit=sample_limit,
        )
    if json_path is not None:
        outputs["json"] = export_report_json(
            report,
            json_path,
            charts=charts,
            title=title,
            sample_limit=sample_limit,
            indent=indent,
        )
    if summary_path is not None:
        outputs["summary"] = export_summary_json(
            report, summary_path, run_id=run_id, indent=indent
        )
    return outputs
