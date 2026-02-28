"""HTML export utilities."""

from __future__ import annotations

import html
from pathlib import Path
from collections.abc import Mapping, MutableMapping, Sequence

from themis.core import entities as core_entities
from themis.experiment import orchestrator
from themis.experiment.export._shared import _collect_sample_metadata, ChartLike


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
    cost_section = _render_cost_section(report)
    metrics_table = _render_metric_table(report)
    samples_table = _render_sample_table(
        report,
        metadata_by_sample,
        metadata_fields,
        metric_names,
        limit=sample_limit,
    )
    chart_sections = "\n".join(_render_chart_section(chart) for chart in charts or ())
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
    .cost-highlight {{ color: #059669; font-size: 1.2rem; font-weight: 600; }}
    .cost-section {{ background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(15,23,42,0.08); margin-bottom: 1.5rem; }}
    .cost-section h2 {{ font-size: 1.3rem; margin-top: 0; margin-bottom: 1rem; }}
    .cost-section h3 {{ font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {summary_section}
  {cost_section}
  {metrics_table}
  {chart_sections}
  {samples_table}
</body>
</html>"""
    return html_doc


def _row_from_evaluation_record(
    record: core_entities.EvaluationRecord,
    *,
    metadata_by_sample: Mapping[str, MutableMapping[str, object]],
    metadata_fields: Sequence[str],
    metric_names: Sequence[str],
    include_failures: bool,
) -> dict[str, object]:
    sample_id = record.sample_id or ""

    # Generate the same condition ID used in _collect_sample_metadata
    # We need to map back to the GenerationRecord that created this EvaluationRecord
    # This is a workaround since we need access to the original task details

    # Create a mapping function to find the corresponding generation record
    # For now, we'll use a simple heuristic based on the available data
    # In a real implementation, this mapping would need to be passed in

    # Try to extract condition info from the record's metadata
    # This is a hack - ideally we'd pass the original task or generation record
    condition_metadata = {}
    for score in record.scores:
        if hasattr(score, "metadata") and score.metadata:
            condition_metadata.update(score.metadata)

    prompt_template = condition_metadata.get("prompt_template", "unknown")
    model_identifier = condition_metadata.get("model_identifier", "unknown")
    sampling_temp = condition_metadata.get("sampling_temperature", 0.0)
    sampling_max_tokens = condition_metadata.get("sampling_max_tokens", 100)

    condition_id = f"{sample_id}_{prompt_template}_{model_identifier}_{sampling_temp}_{sampling_max_tokens}"

    metadata = metadata_by_sample.get(condition_id, {})
    row: dict[str, object] = {"sample_id": sample_id}
    for field in metadata_fields:
        row[field] = metadata.get(field, "")
    score_by_name = {score.metric_name: score.value for score in record.scores}
    for name in metric_names:
        row[f"metric:{name}"] = score_by_name.get(name, "")
    if include_failures:
        row["failures"] = "; ".join(record.failures)
    return row


def _render_summary(report: orchestrator.ExperimentReport) -> str:
    # Filter out cost from main summary (we'll show it separately)
    metadata_items = sorted((k, v) for k, v in report.metadata.items() if k != "cost")
    failures = len(report.failures)
    metadata_html = "\n".join(
        f'<li class="summary-item"><strong>{html.escape(str(key))}</strong><br /><span class="subtle">{html.escape(str(value))}</span></li>'
        for key, value in metadata_items
    )
    failure_block = f'<li class="summary-item"><strong>Run failures</strong><br /><span class="subtle">{failures}</span></li>'
    return f'<section><h2>Summary</h2><ul class="summary-list">{metadata_html}{failure_block}</ul></section>'


def _render_cost_section(report: orchestrator.ExperimentReport) -> str:
    """Render cost breakdown section if cost data is available."""
    cost_data = report.metadata.get("cost")
    if not cost_data or not isinstance(cost_data, dict):
        return ""

    total_cost = cost_data.get("total_cost", 0.0)
    generation_cost = cost_data.get("generation_cost", 0.0)
    evaluation_cost = cost_data.get("evaluation_cost", 0.0)
    currency = cost_data.get("currency", "USD")
    token_counts = cost_data.get("token_counts", {})
    per_model_costs = cost_data.get("per_model_costs", {})
    api_calls = cost_data.get("api_calls", 0)

    # Main cost summary
    cost_items = [
        f'<li class="summary-item"><strong>Total Cost</strong><br /><span class="cost-highlight">${total_cost:.4f} {currency}</span></li>',
        f'<li class="summary-item"><strong>Generation</strong><br /><span class="subtle">${generation_cost:.4f}</span></li>',
        f'<li class="summary-item"><strong>Evaluation</strong><br /><span class="subtle">${evaluation_cost:.4f}</span></li>',
        f'<li class="summary-item"><strong>API Calls</strong><br /><span class="subtle">{api_calls}</span></li>',
    ]

    # Token counts
    if token_counts:
        prompt_tokens = token_counts.get("prompt_tokens", 0)
        completion_tokens = token_counts.get("completion_tokens", 0)
        total_tokens = token_counts.get("total_tokens", 0)
        cost_items.append(
            f'<li class="summary-item"><strong>Tokens</strong><br />'
            f'<span class="subtle">{total_tokens:,} total ({prompt_tokens:,} prompt + {completion_tokens:,} completion)</span></li>'
        )

    cost_summary = "\n".join(cost_items)

    # Per-model breakdown if available
    model_breakdown = ""
    if per_model_costs:
        model_rows = []
        for model, cost in sorted(
            per_model_costs.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            model_rows.append(
                f"<tr><td>{html.escape(model)}</td><td>${cost:.4f}</td><td>{percentage:.1f}%</td></tr>"
            )
        model_table = "\n".join(model_rows)
        model_breakdown = f"""
        <h3>Cost by Model</h3>
        <table>
            <thead>
                <tr><th>Model</th><th>Cost</th><th>% of Total</th></tr>
            </thead>
            <tbody>
                {model_table}
            </tbody>
        </table>
        """

    return f"""
    <section>
        <h2>💰 Cost Breakdown</h2>
        <ul class="summary-list">
            {cost_summary}
        </ul>
        {model_breakdown}
    </section>
    """


def _render_metric_table(report: orchestrator.ExperimentReport) -> str:
    rows = []
    for name in sorted(report.evaluation_report.metrics.keys()):
        metric = report.evaluation_report.metrics[name]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{metric.count}</td><td>{metric.mean:.4f}</td></tr>"
        )
    table_body = "\n".join(rows) or '<tr><td colspan="3">No metrics recorded</td></tr>'
    return (
        "<section><h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Count"
        "</th><th>Mean</th></tr></thead><tbody>"
        + table_body
        + "</tbody></table></section>"
    )


def _render_sample_table(
    report: orchestrator.ExperimentReport,
    metadata_by_sample: Mapping[str, MutableMapping[str, object]],
    metadata_fields: Sequence[str],
    metric_names: Sequence[str],
    *,
    limit: int,
) -> str:
    head_cells = [
        "sample_id",
        *metadata_fields,
        *[f"metric:{name}" for name in metric_names],
    ]
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
            f'<tr><td colspan="{len(head_cells) + 1}">No evaluation records</td></tr>'
        )
    footer = ""
    if len(report.evaluation_report.records) > limit:
        remaining = len(report.evaluation_report.records) - limit
        footer = f'<p class="subtle">Showing first {limit} rows ({remaining} more not rendered).</p>'
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
            f'<section class="chart-section"><h3 class="chart-title">{html.escape(chart.title)}</h3>'
            '<p class="subtle">No data points</p></section>'
        )
    svg_markup = _chart_to_svg(chart)
    rows = "\n".join(
        f"<tr><td>{html.escape(point.label)}</td><td>{html.escape(str(point.x_value))}</td>"
        f"<td>{point.metric_value:.4f}</td><td>{point.count}</td></tr>"
        for point in chart.points
    )
    table = (
        '<table class="chart-table"><thead><tr><th>Label</th><th>X value</th><th>Metric'
        "</th><th>Count</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )
    return (
        f'<section class="chart-section"><h3 class="chart-title">{html.escape(chart.title)}</h3>'
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
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb"></circle>'
        for x, y in zip(x_positions, y_positions)
    )
    labels = "\n".join(
        f'<text x="{x:.2f}" y="{height - margin / 4:.2f}" text-anchor="middle" font-size="12">{html.escape(point.label)}</text>'
        for x, point in zip(x_positions, chart.points)
    )
    y_labels = (
        f'<text x="{margin / 2:.2f}" y="{height - margin:.2f}" font-size="12">{min_value:.2f}</text>'
        f'<text x="{margin / 2:.2f}" y="{margin:.2f}" font-size="12">{max_value:.2f}</text>'
    )
    axis_lines = (
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8" />'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8" />'
    )
    polyline_markup = (
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{polyline}"></polyline>'
        if count > 1
        else ""
    )
    return (
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(chart.metric_name)} vs {html.escape(chart.x_label)}">'
        + axis_lines
        + polyline_markup
        + circles
        + labels
        + y_labels
        + "</svg>"
    )
