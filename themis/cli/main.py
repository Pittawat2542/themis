"""Simplified CLI for Themis - Five core commands only.

This is the new unified CLI that leverages the themis.evaluate() API.
It replaces 20+ commands with 5 essential ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Sequence

from cyclopts import App, Parameter

# Import provider modules to ensure they register themselves
try:
    from themis.generation import clients  # noqa: F401 - registers fake provider
    from themis.generation.providers import (
        litellm_provider,  # noqa: F401
        vllm_provider,  # noqa: F401
    )
except ImportError:
    pass

app = App(
    name="themis",
    help="Dead simple LLM evaluation platform",
    version="2.0.0-alpha.1",
)


@app.command
def eval(
    benchmark_or_dataset: Annotated[str, Parameter(name="BENCHMARK_OR_DATASET", show_default=False)],
    *,
    model: Annotated[str, Parameter(help="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")],
    limit: Annotated[int | None, Parameter(help="Maximum number of samples")] = None,
    prompt: Annotated[str | None, Parameter(help="Custom prompt template")] = None,
    temperature: Annotated[float, Parameter(help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, Parameter(help="Maximum tokens to generate")] = 512,
    storage: Annotated[str | None, Parameter(help="Storage location (local path or s3://...)")] = None,
    run_id: Annotated[str | None, Parameter(help="Unique run identifier")] = None,
    resume: Annotated[bool, Parameter(help="Resume from cached results")] = True,
    distributed: Annotated[bool, Parameter(help="Use distributed execution with Ray")] = False,
    workers: Annotated[int, Parameter(help="Number of workers for distributed execution")] = 4,
    output: Annotated[str | None, Parameter(help="Output file (CSV, JSON, or HTML)")] = None,
) -> int:
    """Run an evaluation on a benchmark or custom dataset.
    
    Examples:
        # Simple benchmark
        themis eval math500 --model gpt-4 --limit 100
        
        # Custom dataset
        themis eval data.jsonl --model claude-3-opus --prompt "Q: {question}\\nA:"
        
        # Distributed execution
        themis eval gsm8k --model gpt-4 --distributed --workers 8
    """
    import themis
    from themis.experiment import export as experiment_export
    
    print(f"Running evaluation: {benchmark_or_dataset}")
    print(f"Model: {model}")
    if limit:
        print(f"Limit: {limit} samples")
    print()
    
    # Check if it's a file (custom dataset)
    if Path(benchmark_or_dataset).exists():
        print(f"Loading custom dataset from: {benchmark_or_dataset}")
        # TODO: Load dataset from file
        print("Error: Custom dataset files not yet implemented")
        return 1
    
    try:
        # Run evaluation using unified API
        report = themis.evaluate(
            benchmark_or_dataset,
            model=model,
            limit=limit,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            storage=storage,
            run_id=run_id,
            resume=resume,
            distributed=distributed,
            workers=workers,
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        # Print metrics
        eval_report = report.evaluation_report
        if eval_report and eval_report.aggregates:
            print("\nMetrics:")
            for agg in eval_report.aggregates:
                print(f"  {agg.metric_name}: {agg.mean:.4f} (±{agg.std:.4f})")
        
        # Print sample counts
        total = len(report.generation_results)
        failures = len(report.failures)
        successful = total - failures
        print(f"\nSamples: {successful}/{total} successful")
        
        # Export if requested
        if output:
            output_path = Path(output)
            suffix = output_path.suffix.lower()
            
            if suffix == ".csv":
                experiment_export.export_csv(report, output_path)
                print(f"\nExported to CSV: {output_path}")
            elif suffix == ".json":
                experiment_export.export_json(report, output_path)
                print(f"\nExported to JSON: {output_path}")
            elif suffix in [".html", ".htm"]:
                experiment_export.export_html(report, output_path)
                print(f"\nExported to HTML: {output_path}")
            else:
                print(f"\nWarning: Unknown output format: {suffix}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


@app.command
def compare(
    run_ids: Annotated[list[str], Parameter(name="RUN_IDS", show_default=False)],
    *,
    metric: Annotated[str | None, Parameter(help="Metric to compare")] = None,
    output: Annotated[str | None, Parameter(help="Output file (HTML or Markdown)")] = None,
    show_diff: Annotated[bool, Parameter(help="Show examples where results differ")] = False,
) -> int:
    """Compare results from multiple runs with statistical tests.
    
    Performs pairwise comparisons across all specified runs and metrics,
    computing win/loss matrices and statistical significance.
    
    Examples:
        # Compare two runs
        themis compare run-1 run-2
        
        # Compare with specific metric
        themis compare run-1 run-2 run-3 --metric ExactMatch
        
        # Export to HTML
        themis compare run-1 run-2 --output comparison.html --show-diff
    """
    try:
        if len(run_ids) < 2:
            print("Error: Need at least 2 runs to compare", file=sys.stderr)
            return 1
        
        # Determine storage path (default to .cache/experiments)
        storage_path = Path(".cache/experiments")
        
        if not storage_path.exists():
            print(f"Error: Storage path not found: {storage_path}", file=sys.stderr)
            print(f"Tip: Specify storage path with THEMIS_STORAGE env var", file=sys.stderr)
            return 1
        
        # Run comparison
        print(f"Comparing {len(run_ids)} runs: {', '.join(run_ids)}")
        print(f"Storage: {storage_path}")
        print()
        
        from themis.comparison import compare_runs
        from themis.comparison.statistics import StatisticalTest
        
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
        import traceback
        traceback.print_exc()
        return 1


@app.command
def serve(
    *,
    port: Annotated[int, Parameter(help="Port to run server on")] = 8080,
    host: Annotated[str, Parameter(help="Host to bind to")] = "127.0.0.1",
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    reload: Annotated[bool, Parameter(help="Enable auto-reload (dev mode)")] = False,
) -> int:
    """Start the Themis API server with REST and WebSocket endpoints.
    
    Provides:
    - REST API for listing/viewing runs
    - Comparison endpoints with statistical tests
    - WebSocket for real-time monitoring
    - Interactive API docs at /docs
    
    Examples:
        # Start server on default port
        themis serve
        
        # Custom port and storage
        themis serve --port 3000 --storage ~/.themis/runs
        
        # Development mode with auto-reload
        themis serve --reload
    """
    try:
        from themis.server import create_app
        import uvicorn
    except ImportError:
        print("Error: FastAPI server dependencies not installed", file=sys.stderr)
        print("\nInstall with: pip install themis[server]", file=sys.stderr)
        print("           or: uv pip install themis[server]", file=sys.stderr)
        return 1
    
    # Determine storage path
    storage_path = Path(storage) if storage else Path(".cache/experiments")
    
    print(f"Starting Themis API server...")
    print(f"  URL:     http://{host}:{port}")
    print(f"  Storage: {storage_path}")
    print(f"  Docs:    http://{host}:{port}/docs")
    print()
    
    # Create app
    app_instance = create_app(storage_path=storage_path)
    
    # Run server
    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    
    return 0


@app.command
def list(
    what: Annotated[str, Parameter(name="WHAT", show_default=False)],
    *,
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    limit: Annotated[int | None, Parameter(help="Limit number of results")] = None,
) -> int:
    """List runs, benchmarks, or available metrics.
    
    Args:
        what: What to list (runs, benchmarks, or metrics)
    
    Examples:
        # List all runs
        themis list runs
        
        # List available benchmarks
        themis list benchmarks
        
        # List available metrics
        themis list metrics
    """
    # Validate input
    if what not in ["runs", "benchmarks", "metrics"]:
        print(f"Error: '{what}' is not valid. Choose from: runs, benchmarks, metrics")
        return 1
    
    if what == "benchmarks":
        from themis.presets import list_benchmarks
        
        benchmarks = list_benchmarks()
        print("Available benchmarks:")
        for benchmark in benchmarks:
            print(f"  - {benchmark}")
        return 0
        
    elif what == "metrics":
        print("Available metrics:")
        print("  Math:")
        print("    - exact_match")
        print("    - math_verify")
        print("  General:")
        print("    - response_length")
        print("\n  Note: NLP and code metrics will be added in Phase 2")
        return 0
        
    elif what == "runs":
        print("Listing runs...")
        print("Note: Run listing not yet fully implemented")
        return 1
    
    return 0


@app.command
def clean(
    *,
    storage: Annotated[str | None, Parameter(help="Storage path to clean")] = None,
    older_than: Annotated[int | None, Parameter(help="Remove runs older than N days")] = None,
    dry_run: Annotated[bool, Parameter(help="Show what would be deleted")] = False,
) -> int:
    """Clean up old runs and cached data.
    
    Examples:
        # Dry run to see what would be deleted
        themis clean --dry-run
        
        # Remove runs older than 30 days
        themis clean --older-than 30
    """
    print("Cleaning storage...")
    print("Note: Storage cleanup not yet implemented")
    print("This will be implemented in Phase 6")
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
    <p><strong>Runs:</strong> {', '.join(report.run_ids)}</p>
    <p><strong>Metrics:</strong> {', '.join(report.metrics)}</p>
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
    lines = []
    lines.append("# Comparison Report")
    lines.append("")
    lines.append(f"**Runs:** {', '.join(report.run_ids)}")
    lines.append(f"**Metrics:** {', '.join(report.metrics)}")
    lines.append(f"**Overall Best:** {report.overall_best_run}")
    lines.append("")
    
    lines.append("## Best Run Per Metric")
    lines.append("")
    for metric, run_id in report.best_run_per_metric.items():
        lines.append(f"- **{metric}:** {run_id}")
    lines.append("")
    
    lines.append("## Win/Loss Matrices")
    lines.append("")
    for metric, matrix in report.win_loss_matrices.items():
        lines.append(f"### {metric}")
        lines.append("")
        lines.append(matrix.to_table())
        lines.append("")
    
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parsed_argv = list(argv) if argv is not None else None
    try:
        result = app(parsed_argv)
    except SystemExit as exc:  # pragma: no cover - CLI integration path
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
