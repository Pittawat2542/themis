"""Comparison commands."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.cli.utils import resolve_storage_root
from themis.utils import logging_utils


def compare_command(
    run_ids: Annotated[
        builtins.list[str], Parameter(name="RUN_IDS", show_default=False)
    ],
    *,
    metric: Annotated[str | None, Parameter(help="Metric to compare")] = None,
    storage: Annotated[
        str | None, Parameter(help="Storage location (local path or s3://...)")
    ] = None,
    output: Annotated[
        str | None, Parameter(help="Output file (HTML or Markdown)")
    ] = None,
    show_diff: Annotated[
        bool, Parameter(help="Show examples where results differ")
    ] = False,
    verbose: Annotated[bool, Parameter(help="Enable debug logging")] = False,
    json_logs: Annotated[bool, Parameter(help="Output logs as JSON")] = False,
) -> int:
    """Compare results from multiple runs with statistical tests."""
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )
    try:
        if len(run_ids) < 2:
            print("Error: Need at least 2 runs to compare", file=sys.stderr)
            return 1

        # Determine storage path (default to .cache/experiments)
        storage_path = resolve_storage_root(storage)

        if not storage_path.exists():
            print(f"Error: Storage path not found: {storage_path}", file=sys.stderr)
            print(
                "Tip: Specify storage path with THEMIS_STORAGE env var", file=sys.stderr
            )
            return 1

        # Run comparison
        print(f"Comparing {len(run_ids)} runs: {', '.join(run_ids)}")
        print(f"Storage: {storage_path}")
        print()

        from themis.experiment.comparison import compare_runs
        from themis.evaluation.statistics.comparison_tests import StatisticalTest

        metrics_list = [metric] if metric else None

        report = compare_runs(
            run_ids=run_ids,
            storage_path=storage_path,
            metrics=metrics_list,
            statistical_test=StatisticalTest.BOOTSTRAP,
            alpha=0.05,
        )

        # Print summary
        print(report.summary(include_details=show_diff))

        # Export if requested
        if output:
            output_path = Path(output)
            suffix = output_path.suffix.lower()

            if suffix == ".json":
                import json

                output_path.write_text(json.dumps(report.to_dict(), indent=2))
                print(f"\n✓ Exported to JSON: {output_path}")
            elif suffix == ".html":
                html = _generate_comparison_html(report)
                output_path.write_text(html)
                print(f"\n✓ Exported to HTML: {output_path}")
            elif suffix == ".md":
                md = _generate_comparison_markdown(report)
                output_path.write_text(md)
                print(f"\n✓ Exported to Markdown: {output_path}")
            else:
                print(f"\nWarning: Unknown output format: {suffix}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Command failed", exc_info=True)
        return 1


def _generate_comparison_html(report) -> str:
    """Generate HTML report for comparison."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .win {{ background-color: #d4edda; }}
        .loss {{ background-color: #f8d7da; }}
        .tie {{ background-color: #fff3cd; }}
        .significant {{ font-weight: bold; color: #28a745; }}
    </style>
</head>
<body>
    <h1>Comparison Report</h1>
    <p><strong>Runs:</strong> {", ".join(report.run_ids)}</p>
    <p><strong>Metrics:</strong> {", ".join(report.metrics)}</p>
    <p><strong>Overall Best:</strong> {report.overall_best_run}</p>
    
    <h2>Best Run Per Metric</h2>
    <ul>
"""

    for metric, run_id in report.best_run_per_metric.items():
        html += f"        <li><strong>{metric}:</strong> {run_id}</li>\n"

    html += """    </ul>
    
    <h2>Win/Loss Matrices</h2>
"""

    for metric, matrix in report.win_loss_matrices.items():
        html += f"    <h3>{metric}</h3>\n"
        html += "    <table>\n"
        html += "        <tr><th>Run</th>"
        for rid in matrix.run_ids:
            html += f"<th>{rid}</th>"
        html += "</tr>\n"

        for i, run_id in enumerate(matrix.run_ids):
            html += f"        <tr><td><strong>{run_id}</strong></td>"
            for j in range(len(matrix.run_ids)):
                result = matrix.matrix[i][j]
                css_class = result if result in ["win", "loss", "tie"] else ""
                html += f'<td class="{css_class}">{result}</td>'
            html += "</tr>\n"

        html += "    </table>\n"

    html += """</body>
</html>"""

    return html


def _generate_comparison_markdown(report) -> str:
    """Generate Markdown report for comparison."""
    md = "# Comparison Report\n\n"
    md += f"**Runs:** {', '.join(report.run_ids)}\n"
    md += f"**Metrics:** {', '.join(report.metrics)}\n"
    md += f"**Overall Best:** {report.overall_best_run}\n\n"

    md += "## Best Run Per Metric\n"
    for metric, run_id in report.best_run_per_metric.items():
        md += f"- **{metric}:** {run_id}\n"

    md += "\n## Win/Loss Matrices\n"
    for metric, matrix in report.win_loss_matrices.items():
        md += f"### {metric}\n"

        # Header
        md += "| Run | " + " | ".join(matrix.run_ids) + " |\n"
        md += "| --- | " + " | ".join(["---"] * len(matrix.run_ids)) + " |\n"

        # Rows
        for i, run_id in enumerate(matrix.run_ids):
            row = [f"**{run_id}**"] + matrix.matrix[i]
            md += "| " + " | ".join(row) + " |\n"
        md += "\n"

    return md
