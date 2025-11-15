"""CLI utility functions."""

from __future__ import annotations

from pathlib import Path

from themis.experiment import export as experiment_export
from themis.experiment import orchestrator


def export_outputs(
    report: orchestrator.ExperimentReport,
    *,
    csv_output: Path | None,
    html_output: Path | None,
    json_output: Path | None,
    title: str,
) -> None:
    """Export experiment report to various formats.

    Args:
        report: Experiment report to export
        csv_output: Optional path for CSV export
        html_output: Optional path for HTML export
        json_output: Optional path for JSON export
        title: Title for the report
    """
    outputs = experiment_export.export_report_bundle(
        report,
        csv_path=csv_output,
        html_path=html_output,
        json_path=json_output,
        title=title,
    )
    for kind, output_path in outputs.items():
        print(f"Exported {kind.upper()} to {output_path}")


def effective_total(total: int, limit: int | None) -> int:
    """Calculate effective total based on limit.

    Args:
        total: Total number of items
        limit: Optional limit

    Returns:
        Effective total (min of total and limit)
    """
    if limit is None:
        return total
    return min(total, limit)
