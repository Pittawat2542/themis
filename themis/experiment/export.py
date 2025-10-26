"""Utilities for exporting experiment results to CSV, JSON, and HTML."""

from __future__ import annotations

import csv
import html
import json
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, MutableMapping, Protocol, Sequence

from themis.core import entities as core_entities
from themis.experiment import orchestrator


class ChartPointLike(Protocol):
    label: str
    x_value: object
    metric_value: float
    metric_name: str
    count: int


class ChartLike(Protocol):
    title: str
    x_label: str
    y_label: str
    metric_name: str
    points: Sequence[ChartPointLike]


def export_report_csv(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    include_failures: bool = True,
) -> Path:
    """Write per-sample metrics to a CSV file for offline analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_sample, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )
    metric_names = sorted(report.evaluation_report.metrics.keys())
    fieldnames = ["sample_id"] + metadata_fields + [
        f"metric:{name}" for name in metric_names
    ]
    if include_failures:
        fieldnames.append("failures")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in report.evaluation_report.records:
            row = _row_from_evaluation_record(
                record,
                metadata_by_sample=metadata_by_sample,
                metadata_fields=metadata_fields,
                metric_names=metric_names,
                include_failures=include_failures,
            )
            writer.writerow(row)
    return path


def export_html_report(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int = 100,
) -> Path:
    """Render the experiment report as an HTML document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_html_report(
        report,
        charts=charts,
        title=title,
        sample_limit=sample_limit,
    )
    path.write_text(html_content, encoding="utf-8")
    return path


def export_report_json(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int | None = None,
    indent: int = 2,
) -> Path:
    """Serialize the report details to JSON for downstream tooling."""

    payload = build_json_report(
        report,
        charts=charts,
        title=title,
        sample_limit=sample_limit,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def export_report_bundle(
    report: orchestrator.ExperimentReport,
    *,
    csv_path: str | Path | None = None,
    html_path: str | Path | None = None,
    json_path: str | Path | None = None,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int = 100,
    indent: int = 2,
) -> OrderedDict[str, Path]:
    """Convenience helper that writes multiple export formats at once."""

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
    return outputs


def render_html_report(
    report: orchestrator.ExperimentReport,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int = 100,
) -> str:
    """Return an HTML string summarizing the experiment results."""

    metadata_by_sample, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )
    metric_names = sorted(report.evaluation_report.metrics.keys())
    summary_section = _render_summary(report)
    metrics_table = _render_metric_table(report)
    samples_table = _render_sample_table(
        report,
        metadata_by_sample,
        metadata_fields,
        metric_names,
        limit=sample_limit,
    )
    chart_sections = "\n".join(
        _render_chart_section(chart) for chart in charts or ()
    )
    html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; background: #f6f8fb; color: #1f2933; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
    section {{ margin-bottom: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 2px rgba(15,23,42,0.08); }}
    th, td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e7eb; font-size: 0.95rem; text-align: left; }}
    th {{ background: #f0f2f8; font-weight: 600; }}
    tbody tr:nth-child(odd) {{ background: #fafbff; }}
    .summary-list {{ list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; }}
    .summary-item {{ background: white; padding: 0.75rem 1rem; border-radius: 8px; box-shadow: inset 0 0 0 1px #e5e7eb; }}
    .chart-section {{ background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(15,23,42,0.08); margin-bottom: 1.5rem; }}
    .chart-title {{ margin: 0 0 0.5rem 0; font-size: 1.1rem; }}
    .chart-svg {{ width: 100%; height: 320px; }}
    .chart-table {{ margin-top: 0.75rem; }}
    .subtle {{ color: #6b7280; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {summary_section}
  {metrics_table}
  {chart_sections}
  {samples_table}
</body>
</html>"""
    return html_doc


def build_json_report(
    report: orchestrator.ExperimentReport,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int | None = None,
) -> dict[str, object]:
    metadata_by_sample, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )
    metric_names = sorted(report.evaluation_report.metrics.keys())
    samples = []
    limit = sample_limit if sample_limit is not None else len(
        report.evaluation_report.records
    )
    for index, record in enumerate(report.evaluation_report.records):
        if index >= limit:
            break
        sample_id = record.sample_id or ""
        sample_metadata = dict(metadata_by_sample.get(sample_id, {}))
        scores = [
            {
                "metric": score.metric_name,
                "value": score.value,
                "details": score.details,
                "metadata": score.metadata,
            }
            for score in record.scores
        ]
        samples.append(
            {
                "sample_id": sample_id,
                "metadata": sample_metadata,
                "scores": scores,
                "failures": list(record.failures),
            }
        )

    payload = {
        "title": title,
        "summary": {
            **report.metadata,
            "run_failures": len(report.failures),
            "evaluation_failures": len(report.evaluation_report.failures),
        },
        "metrics": [
            {
                "name": name,
                "count": metric.count,
                "mean": metric.mean,
            }
            for name, metric in sorted(
                report.evaluation_report.metrics.items(), key=lambda item: item[0]
            )
        ],
        "samples": samples,
        "rendered_sample_limit": limit,
        "total_samples": len(report.evaluation_report.records),
        "charts": [chart.as_dict() if hasattr(chart, "as_dict") else _chart_to_dict(chart) for chart in charts or ()],
        "run_failures": [
            {"sample_id": failure.sample_id, "message": failure.message}
            for failure in report.failures
        ],
        "evaluation_failures": [
            {"sample_id": failure.sample_id, "message": failure.message}
            for failure in report.evaluation_report.failures
        ],
        "metrics_rendered": metric_names,
    }
    return payload


def _row_from_evaluation_record(
    record: core_entities.EvaluationRecord,
    *,
    metadata_by_sample: Mapping[str, MutableMapping[str, object]],
    metadata_fields: Sequence[str],
    metric_names: Sequence[str],
    include_failures: bool,
) -> dict[str, object]:
    sample_id = record.sample_id or ""
    metadata = metadata_by_sample.get(sample_id, {})
    row: dict[str, object] = {"sample_id": sample_id}
    for field in metadata_fields:
        row[field] = metadata.get(field, "")
    score_by_name = {score.metric_name: score.value for score in record.scores}
    for name in metric_names:
        row[f"metric:{name}"] = score_by_name.get(name, "")
    if include_failures:
        row["failures"] = "; ".join(record.failures)
    return row


def _collect_sample_metadata(
    records: Sequence[core_entities.GenerationRecord],
) -> tuple[dict[str, MutableMapping[str, object]], list[str]]:
    metadata: dict[str, MutableMapping[str, object]] = {}
    for index, record in enumerate(records):
        sample_id = _extract_sample_id(record.task.metadata)
        if sample_id is None:
            sample_id = f"sample-{index}"
        metadata.setdefault(sample_id, {})
        metadata[sample_id].update(_metadata_from_task(record))
    fields = sorted({field for meta in metadata.values() for field in meta.keys()})
    return metadata, fields


def _extract_sample_id(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("dataset_id") or metadata.get("sample_id")
    if value is None:
        return None
    return str(value)


def _metadata_from_task(record: core_entities.GenerationRecord) -> dict[str, object]:
    metadata = dict(record.task.metadata)
    metadata.setdefault("model_identifier", record.task.model.identifier)
    metadata.setdefault("model_provider", record.task.model.provider)
    metadata.setdefault("prompt_template", record.task.prompt.spec.name)
    metadata.setdefault("sampling_temperature", record.task.sampling.temperature)
    metadata.setdefault("sampling_top_p", record.task.sampling.top_p)
    metadata.setdefault("sampling_max_tokens", record.task.sampling.max_tokens)
    return metadata


def _render_summary(report: orchestrator.ExperimentReport) -> str:
    metadata_items = sorted(report.metadata.items())
    failures = len(report.failures)
    metadata_html = "\n".join(
        f"<li class=\"summary-item\"><strong>{html.escape(str(key))}</strong><br /><span class=\"subtle\">{html.escape(str(value))}</span></li>"
        for key, value in metadata_items
    )
    failure_block = (
        f"<li class=\"summary-item\"><strong>Run failures</strong><br /><span class=\"subtle\">{failures}</span></li>"
    )
    return f"<section><h2>Summary</h2><ul class=\"summary-list\">{metadata_html}{failure_block}</ul></section>"


def _render_metric_table(report: orchestrator.ExperimentReport) -> str:
    rows = []
    for name in sorted(report.evaluation_report.metrics.keys()):
        metric = report.evaluation_report.metrics[name]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{metric.count}</td><td>{metric.mean:.4f}</td></tr>"
        )
    table_body = "\n".join(rows) or "<tr><td colspan=\"3\">No metrics recorded</td></tr>"
    return (
        "<section><h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Count" 
        "</th><th>Mean</th></tr></thead><tbody>" + table_body + "</tbody></table></section>"
    )


def _render_sample_table(
    report: orchestrator.ExperimentReport,
    metadata_by_sample: Mapping[str, MutableMapping[str, object]],
    metadata_fields: Sequence[str],
    metric_names: Sequence[str],
    *,
    limit: int,
) -> str:
    head_cells = ["sample_id", *metadata_fields, *[f"metric:{name}" for name in metric_names]]
    head_html = "".join(f"<th>{html.escape(label)}</th>" for label in head_cells)
    body_rows: list[str] = []
    for index, record in enumerate(report.evaluation_report.records):
        if index >= limit:
            break
        row = _row_from_evaluation_record(
            record,
            metadata_by_sample=metadata_by_sample,
            metadata_fields=metadata_fields,
            metric_names=metric_names,
            include_failures=True,
        )
        cells = [html.escape(str(row.get(label, ""))) for label in head_cells]
        cells.append(html.escape(row.get("failures", "")))
        body_rows.append(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    if not body_rows:
        body_rows.append(
            f"<tr><td colspan=\"{len(head_cells)+1}\">No evaluation records</td></tr>"
        )
    footer = ""
    if len(report.evaluation_report.records) > limit:
        remaining = len(report.evaluation_report.records) - limit
        footer = f"<p class=\"subtle\">Showing first {limit} rows ({remaining} more not rendered).</p>"
    return (
        "<section><h2>Sample breakdown</h2><table><thead><tr>"
        + head_html
        + "<th>failures</th></tr></thead><tbody>"
        + "\n".join(body_rows)
        + "</tbody></table>"
        + footer
        + "</section>"
    )


def _render_chart_section(chart: ChartLike) -> str:
    if not chart.points:
        return (
            f"<section class=\"chart-section\"><h3 class=\"chart-title\">{html.escape(chart.title)}</h3>"
            "<p class=\"subtle\">No data points</p></section>"
        )
    svg_markup = _chart_to_svg(chart)
    rows = "\n".join(
        f"<tr><td>{html.escape(point.label)}</td><td>{html.escape(str(point.x_value))}</td>"
        f"<td>{point.metric_value:.4f}</td><td>{point.count}</td></tr>"
        for point in chart.points
    )
    table = (
        "<table class=\"chart-table\"><thead><tr><th>Label</th><th>X value</th><th>Metric" 
        "</th><th>Count</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )
    return (
        f"<section class=\"chart-section\"><h3 class=\"chart-title\">{html.escape(chart.title)}</h3>"
        + svg_markup
        + table
        + "</section>"
    )


def _chart_to_svg(chart: ChartLike) -> str:
    width, height, margin = 640, 320, 42
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    values = [point.metric_value for point in chart.points]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    count = len(chart.points)
    if count == 1:
        x_positions = [margin + plot_width / 2]
    else:
        step = plot_width / (count - 1)
        x_positions = [margin + index * step for index in range(count)]

    def scale_y(value: float) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return margin + (plot_height * (1 - ratio))

    y_positions = [scale_y(point.metric_value) for point in chart.points]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(x_positions, y_positions))
    circles = "\n".join(
        f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"5\" fill=\"#2563eb\"></circle>"
        for x, y in zip(x_positions, y_positions)
    )
    labels = "\n".join(
        f"<text x=\"{x:.2f}\" y=\"{height - margin / 4:.2f}\" text-anchor=\"middle\" font-size=\"12\">{html.escape(point.label)}</text>"
        for x, point in zip(x_positions, chart.points)
    )
    y_labels = (
        f"<text x=\"{margin/2:.2f}\" y=\"{height - margin:.2f}\" font-size=\"12\">{min_value:.2f}</text>"
        f"<text x=\"{margin/2:.2f}\" y=\"{margin:.2f}\" font-size=\"12\">{max_value:.2f}</text>"
    )
    axis_lines = (
        f"<line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#94a3b8\" />"
        f"<line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#94a3b8\" />"
    )
    polyline_markup = (
        f"<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2\" points=\"{polyline}\"></polyline>"
        if count > 1
        else ""
    )
    return (
        f"<svg class=\"chart-svg\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(chart.metric_name)} vs {html.escape(chart.x_label)}\">"
        + axis_lines
        + polyline_markup
        + circles
        + labels
        + y_labels
        + "</svg>"
    )


def _chart_to_dict(chart: ChartLike) -> dict[str, object]:
    return {
        "title": chart.title,
        "x_label": chart.x_label,
        "y_label": chart.y_label,
        "metric": chart.metric_name,
        "points": [
            {
                "label": point.label,
                "x": getattr(point, "x_value", getattr(point, "x", None)),
                "value": point.metric_value,
                "count": point.count,
            }
            for point in chart.points
        ],
    }


__all__ = [
    "export_report_csv",
    "export_html_report",
    "export_report_json",
    "export_report_bundle",
    "render_html_report",
    "build_json_report",
]
