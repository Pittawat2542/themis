"""Dashboard generator for experiment results, costs, and statistics.

This module provides HTML dashboard generation for visualizing experiment
results, cost breakdowns, and statistical analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from themis.evaluation import reports as eval_reports
from themis.evaluation import statistics as eval_stats
from themis.utils import cost_tracking


def generate_html_dashboard(
    evaluation_report: eval_reports.EvaluationReport,
    cost_summary: cost_tracking.CostSummary | None = None,
    statistical_summaries: Dict[str, eval_stats.StatisticalSummary] | None = None,
    output_path: str | Path = "dashboard.html",
    title: str = "Themis Experiment Dashboard",
) -> None:
    """Generate HTML dashboard with evaluation results, costs, and statistics.

    Args:
        evaluation_report: Evaluation report with metric results
        cost_summary: Optional cost summary
        statistical_summaries: Optional dictionary mapping metric names to statistical summaries
        output_path: Path to output HTML file
        title: Dashboard title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = _generate_html(
        evaluation_report,
        cost_summary,
        statistical_summaries,
        title,
    )

    with open(output_path, "w") as f:
        f.write(html_content)


def _generate_html(
    evaluation_report: eval_reports.EvaluationReport,
    cost_summary: cost_tracking.CostSummary | None,
    statistical_summaries: Dict[str, eval_stats.StatisticalSummary] | None,
    title: str,
) -> str:
    """Generate complete HTML dashboard content."""

    # Build sections
    metrics_section = _build_metrics_section(evaluation_report.metrics)

    stats_section = ""
    if statistical_summaries:
        stats_section = _build_statistics_section(statistical_summaries)

    cost_section = ""
    if cost_summary:
        cost_section = _build_cost_section(cost_summary)

    failures_section = ""
    if evaluation_report.failures:
        failures_section = _build_failures_section(evaluation_report.failures)

    # Compose full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }}
        
        .metric-name {{
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            color: #3498db;
            margin: 10px 0;
        }}
        
        .metric-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        
        .detail-item {{
            background: white;
            padding: 10px;
            border-radius: 4px;
        }}
        
        .detail-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .detail-value {{
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 4px;
        }}
        
        .cost-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .cost-total {{
            font-size: 48px;
            font-weight: 700;
            margin: 10px 0;
        }}
        
        .cost-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .cost-item {{
            background: rgba(255, 255, 255, 0.1);
            padding: 12px;
            border-radius: 4px;
        }}
        
        .cost-item-name {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .cost-item-value {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 4px;
        }}
        
        .failures {{
            background: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            border-radius: 4px;
        }}
        
        .failure-item {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        
        .confidence-interval {{
            font-size: 14px;
            color: #7f8c8d;
            font-family: 'Courier New', monospace;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        {metrics_section}
        
        {stats_section}
        
        {cost_section}
        
        {failures_section}
    </div>
</body>
</html>"""

    return html


def _build_metrics_section(metrics: Dict[str, eval_reports.MetricAggregate]) -> str:
    """Build metrics overview section."""
    if not metrics:
        return "<p>No metrics available.</p>"

    cards = []
    for metric_name, aggregate in metrics.items():
        card = f"""
        <div class="metric-card">
            <div class="metric-name">{metric_name}</div>
            <div class="metric-value">{aggregate.mean:.4f}</div>
            <div class="metric-details">
                <div class="detail-item">
                    <div class="detail-label">Samples</div>
                    <div class="detail-value">{aggregate.count}</div>
                </div>
            </div>
        </div>
        """
        cards.append(card)

    return f"""
    <h2>📊 Metrics Overview</h2>
    {"".join(cards)}
    """


def _build_statistics_section(
    statistical_summaries: Dict[str, eval_stats.StatisticalSummary],
) -> str:
    """Build detailed statistics section."""
    if not statistical_summaries:
        return ""

    cards = []
    for metric_name, summary in statistical_summaries.items():
        ci_text = ""
        if summary.confidence_interval_95:
            ci = summary.confidence_interval_95
            ci_text = f"""
            <div class="detail-item">
                <div class="detail-label">95% CI</div>
                <div class="detail-value confidence-interval">
                    [{ci.lower:.4f}, {ci.upper:.4f}]
                </div>
            </div>
            """

        card = f"""
        <div class="metric-card">
            <div class="metric-name">{metric_name} - Statistical Analysis</div>
            <div class="metric-details">
                <div class="detail-item">
                    <div class="detail-label">Mean</div>
                    <div class="detail-value">{summary.mean:.4f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Std Dev</div>
                    <div class="detail-value">{summary.std:.4f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Median</div>
                    <div class="detail-value">{summary.median:.4f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Min</div>
                    <div class="detail-value">{summary.min_value:.4f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Max</div>
                    <div class="detail-value">{summary.max_value:.4f}</div>
                </div>
                {ci_text}
            </div>
        </div>
        """
        cards.append(card)

    return f"""
    <h2>📈 Statistical Analysis</h2>
    {"".join(cards)}
    """


def _build_cost_section(cost_summary: cost_tracking.CostSummary) -> str:
    """Build cost tracking section."""

    # Model breakdown
    model_items = []
    for model, cost in sorted(
        cost_summary.cost_by_model.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = (
            (cost / cost_summary.total_cost * 100) if cost_summary.total_cost > 0 else 0
        )
        model_items.append(f"""
        <div class="cost-item">
            <div class="cost-item-name">{model}</div>
            <div class="cost-item-value">${cost:.4f} ({pct:.1f}%)</div>
        </div>
        """)

    # Provider breakdown
    provider_items = []
    for provider, cost in sorted(
        cost_summary.cost_by_provider.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = (
            (cost / cost_summary.total_cost * 100) if cost_summary.total_cost > 0 else 0
        )
        provider_items.append(f"""
        <div class="cost-item">
            <div class="cost-item-name">{provider}</div>
            <div class="cost-item-value">${cost:.4f} ({pct:.1f}%)</div>
        </div>
        """)

    return f"""
    <h2>💰 Cost Tracking</h2>
    <div class="cost-summary">
        <h3 style="color: white; margin-top: 0;">Total Cost</h3>
        <div class="cost-total">${cost_summary.total_cost:.4f}</div>
        <div class="cost-breakdown">
            <div class="cost-item">
                <div class="cost-item-name">Total Tokens</div>
                <div class="cost-item-value">{cost_summary.total_tokens:,}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-name">Input Tokens</div>
                <div class="cost-item-value">{cost_summary.total_input_tokens:,}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-name">Output Tokens</div>
                <div class="cost-item-value">{cost_summary.total_output_tokens:,}</div>
            </div>
            <div class="cost-item">
                <div class="cost-item-name">API Requests</div>
                <div class="cost-item-value">{cost_summary.num_requests:,}</div>
            </div>
        </div>
    </div>
    
    <h3>Cost by Model</h3>
    <div class="cost-breakdown">
        {"".join(model_items)}
    </div>
    
    <h3>Cost by Provider</h3>
    <div class="cost-breakdown">
        {"".join(provider_items)}
    </div>
    """


def _build_failures_section(failures: List[eval_reports.EvaluationFailure]) -> str:
    """Build failures section."""
    if not failures:
        return ""

    failure_items = []
    for failure in failures[:20]:  # Limit to first 20 failures
        sample_id = failure.sample_id or "Unknown"
        failure_items.append(f"""
        <div class="failure-item">
            <strong>Sample: {sample_id}</strong><br>
            {failure.message}
        </div>
        """)

    more_text = ""
    if len(failures) > 20:
        more_text = f"<p><em>...and {len(failures) - 20} more failures</em></p>"

    return f"""
    <h2>⚠️ Failures ({len(failures)})</h2>
    <div class="failures">
        {"".join(failure_items)}
        {more_text}
    </div>
    """


__all__ = ["generate_html_dashboard"]
