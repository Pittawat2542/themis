"""Simplified CLI for Themis - seven focused commands.

This is the unified CLI that leverages the themis.evaluate() API.
It replaces 20+ commands with a smaller, task-oriented set.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Sequence

from cyclopts import App, Parameter
from themis._version import __version__
from themis.utils import logging_utils

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
    version=__version__,
)


@app.command
def demo(
    *,
    model: Annotated[str, Parameter(help="Model identifier")] = "fake-math-llm",
    limit: Annotated[int, Parameter(help="Maximum number of samples")] = 10,
    verbose: Annotated[bool, Parameter(help="Enable debug logging")] = False,
    json_logs: Annotated[bool, Parameter(help="Output logs as JSON")] = False,
) -> int:
    """Run the built-in demo benchmark."""
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )
    return eval(
        "demo",
        model=model,
        limit=limit,
        verbose=verbose,
        json_logs=json_logs,
    )


@app.command
def eval(
    benchmark_or_dataset: Annotated[
        str, Parameter(name="BENCHMARK_OR_DATASET", show_default=False)
    ],
    *,
    model: Annotated[
        str, Parameter(help="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    ],
    limit: Annotated[int | None, Parameter(help="Maximum number of samples")] = None,
    prompt: Annotated[str | None, Parameter(help="Custom prompt template")] = None,
    temperature: Annotated[float, Parameter(help="Sampling temperature")] = 0.0,
    max_tokens: Annotated[int, Parameter(help="Maximum tokens to generate")] = 512,
    storage: Annotated[
        str | None, Parameter(help="Storage location (local path or s3://...)")
    ] = None,
    run_id: Annotated[str | None, Parameter(help="Unique run identifier")] = None,
    resume: Annotated[bool, Parameter(help="Resume from cached results")] = True,
    workers: Annotated[int, Parameter(help="Number of generation workers")] = 4,
    output: Annotated[
        str | None, Parameter(help="Output file (CSV, JSON, or HTML)")
    ] = None,
    verbose: Annotated[bool, Parameter(help="Enable debug logging")] = False,
    json_logs: Annotated[bool, Parameter(help="Output logs as JSON")] = False,
) -> int:
    """Run an evaluation on a benchmark or custom dataset.

    Examples:
        # Simple benchmark
        themis eval math500 --model gpt-4 --limit 100

        # Custom dataset
        themis eval data.jsonl --model claude-3-opus --prompt "Q: {question}\\nA:"

        # Increase generation parallelism
        themis eval gsm8k --model gpt-4 --workers 8
    """
    from themis.experiment import export as experiment_export

    # Configure logging if not already configured (demo check might be redundant but safe)
    # Note: configure_logging uses force=True so last call wins.
    # If called from demo(), it's called twice. First with demo args, then here.
    # We should trust the caller (demo) or ensuring eval sets it if called directly.
    # Since eval is also a command, it needs to set it.
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )

    print(f"Running evaluation: {benchmark_or_dataset}")
    print(f"Model: {model}")
    if limit:
        print(f"Limit: {limit} samples")
    print()

    try:
        custom_dataset_path = Path(benchmark_or_dataset)
        is_custom_dataset = custom_dataset_path.exists()
        storage_root = _resolve_storage_root(storage)

        if is_custom_dataset:
            print(f"Loading custom dataset from: {custom_dataset_path}")
            from themis.api import evaluate as evaluate_api

            dataset_rows, detected_prompt_field, _ = _load_custom_dataset_file(
                custom_dataset_path
            )
            prompt_template = prompt or f"{{{detected_prompt_field}}}"
            report = evaluate_api(
                dataset_rows,
                model=model,
                prompt=prompt_template,
                temperature=temperature,
                max_tokens=max_tokens,
                workers=workers,
                storage=storage_root,
                run_id=run_id,
                resume=resume,
                limit=limit,
            )
        else:
            from themis.evaluation.pipeline import EvaluationPipeline
            from themis.generation.templates import PromptTemplate
            from themis.presets import get_benchmark_preset
            from themis.session import ExperimentSession
            from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

            # Resolve benchmark preset
            preset = get_benchmark_preset(benchmark_or_dataset)

            dataset = preset.load_dataset(limit=limit)

            if prompt is None:
                prompt_template = preset.prompt_template
            else:
                prompt_template = PromptTemplate(name="custom", template=prompt)

            pipeline = EvaluationPipeline(
                extractor=preset.extractor,
                metrics=preset.metrics,
            )

            spec = ExperimentSpec(
                dataset=dataset,
                prompt=prompt_template.template,
                model=model,
                sampling={"temperature": temperature, "max_tokens": max_tokens},
                pipeline=pipeline,
                run_id=run_id,
            )

            report = ExperimentSession().run(
                spec,
                execution=ExecutionSpec(workers=workers),
                storage=StorageSpec(path=storage_root, cache=resume),
            )

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        # Print metrics
        eval_report = report.evaluation_report
        if eval_report:
            print("\nMetrics:")
            if getattr(eval_report, "aggregates", None):
                for agg in eval_report.aggregates:
                    std = getattr(agg, "std", None)
                    if std is None:
                        print(f"  {agg.metric_name}: {agg.mean:.4f}")
                    else:
                        print(f"  {agg.metric_name}: {agg.mean:.4f} (±{std:.4f})")
            elif getattr(eval_report, "metrics", None):
                for name, agg in sorted(eval_report.metrics.items()):
                    print(f"  {name}: {agg.mean:.4f} (n={agg.count})")

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
                experiment_export.export_report_csv(report, output_path)
                print(f"\nExported to CSV: {output_path}")
            elif suffix == ".json":
                experiment_export.export_report_json(report, output_path)
                print(f"\nExported to JSON: {output_path}")
            elif suffix in [".html", ".htm"]:
                experiment_export.export_html_report(report, output_path)
                print(f"\nExported to HTML: {output_path}")
            else:
                print(f"\nWarning: Unknown output format: {suffix}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Command failed", exc_info=True)
        return 1


@app.command
def compare(
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
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )
    try:
        if len(run_ids) < 2:
            print("Error: Need at least 2 runs to compare", file=sys.stderr)
            return 1

        # Determine storage path (default to .cache/experiments)
        storage_path = _resolve_storage_root(storage)

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
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Command failed", exc_info=True)
        return 1


@app.command
def share(
    run_id: Annotated[str, Parameter(name="RUN_ID", show_default=False)],
    *,
    storage: Annotated[
        str | None, Parameter(help="Storage location (defaults to .cache/experiments)")
    ] = None,
    metric: Annotated[
        str | None, Parameter(help="Metric to highlight (default: first available)")
    ] = None,
    output_dir: Annotated[
        Path, Parameter(help="Directory to write share assets")
    ] = Path("."),
) -> int:
    """Generate a shareable results badge + Markdown snippet for a run.

    Examples:
        # Create share assets in current directory
        themis share run-20260118-032014

        # Highlight a specific metric
        themis share run-20260118-032014 --metric accuracy

        # Write to a dedicated folder
        themis share run-20260118-032014 --output-dir share
    """
    from themis.experiment.share import create_share_pack

    storage_root = Path(storage) if storage else Path(".cache/experiments")
    if not storage_root.exists():
        print(f"Error: Storage path not found: {storage_root}", file=sys.stderr)
        return 1

    try:
        share_pack = create_share_pack(
            run_id=run_id,
            storage_root=storage_root,
            output_dir=output_dir,
            metric=metric,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Share failed", exc_info=True)
        return 1

    print("✓ Share assets created")
    print(f"  SVG: {share_pack.svg_path}")
    print(f"  Markdown: {share_pack.markdown_path}")
    print("\nSnippet:")
    print(share_pack.markdown_snippet)
    if share_pack.event_log_path:
        print(f"\nEvent logged to: {share_pack.event_log_path}")
    return 0


@app.command
def serve(
    *,
    port: Annotated[int, Parameter(help="Port to run server on")] = 8080,
    host: Annotated[str, Parameter(help="Host to bind to")] = "127.0.0.1",
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    reload: Annotated[bool, Parameter(help="Enable auto-reload (dev mode)")] = False,
    verbose: Annotated[bool, Parameter(help="Enable debug logging")] = False,
    json_logs: Annotated[bool, Parameter(help="Output logs as JSON")] = False,
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
    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="json" if json_logs else "human",
    )
    try:
        from themis.server import create_app
        import uvicorn
    except ImportError:
        print("Error: FastAPI server dependencies not installed", file=sys.stderr)
        print("\nInstall with: pip install themis[server]", file=sys.stderr)
        print("           or: uv pip install themis[server]", file=sys.stderr)
        return 1

    # Determine storage path
    storage_path = _resolve_storage_root(storage)

    print("Starting Themis API server...")
    print(f"  URL:     http://{host}:{port}")
    print(f"  Storage: {storage_path}")
    print(f"  Docs:    http://{host}:{port}/docs")
    print()

    # Create app
    app_instance = create_app(storage_path=storage_path)

    # Run server
    try:
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except Exception:
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Server failed", exc_info=True)
        return 1

    return 0


@app.command(name="list")
def list_command(
    what: Annotated[str, Parameter(name="WHAT", show_default=False)],
    *,
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    limit: Annotated[int | None, Parameter(help="Limit number of results")] = None,
    verbose: Annotated[bool, Parameter(help="Show detailed information")] = False,
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

    logging_utils.configure_logging(
        level="debug" if verbose else "info",
        log_format="human",
    )

    if what == "benchmarks":
        from themis.presets import get_benchmark_preset, list_benchmarks

        benchmarks = list_benchmarks()
        if limit is not None:
            benchmarks = benchmarks[:limit]
        print("Available benchmarks:")
        for benchmark in benchmarks:
            if verbose:
                preset = get_benchmark_preset(benchmark)
                description = preset.description or "No description"
                print(f"  - {benchmark}: {description}")
            else:
                print(f"  - {benchmark}")
        return 0

    elif what == "metrics":
        print("Available metrics:")
        print("  Core:")
        print("    - exact_match (no extra dependencies)")
        print("    - response_length (no extra dependencies)")
        print("  Math:")
        print("    - math_verify (requires: themis-eval[math], math-verify)")
        print("  NLP (requires: themis-eval[nlp]):")
        print("    - bleu (sacrebleu)")
        print("    - rouge1 / rouge2 / rougeL (rouge-score)")
        print("    - bertscore (bert-score)")
        print("    - meteor (nltk)")
        print("  Code:")
        print("    - pass_at_k (no extra dependencies)")
        print("    - execution_accuracy (no extra dependencies)")
        print("    - codebleu (requires: themis-eval[code], codebleu)")
        print("\nInstall extras: pip install themis-eval[math,nlp,code]")
        return 0

    elif what == "runs":
        from themis.storage import ExperimentStorage

        storage_root = _resolve_storage_root(storage)
        if not storage_root.exists():
            print(f"No storage found at {storage_root}")
            return 1

        storage_backend = ExperimentStorage(storage_root)
        runs = storage_backend.list_runs(limit=limit)
        if not runs:
            print("No runs found.")
            return 0

        print("Runs:")
        for run in runs:
            status = (
                run.status.value if hasattr(run.status, "value") else str(run.status)
            )
            if verbose:
                print(
                    f"  - {run.run_id} [{status}] samples={run.total_samples} "
                    f"created={run.created_at}"
                )
            else:
                print(f"  - {run.run_id}")
        return 0

    return 0


@app.command
def clean(
    *,
    storage: Annotated[str | None, Parameter(help="Storage path to clean")] = None,
    older_than: Annotated[
        int | None, Parameter(help="Remove runs older than N days")
    ] = None,
    dry_run: Annotated[bool, Parameter(help="Show what would be deleted")] = False,
) -> int:
    """Clean up old runs and cached data.

    Examples:
        # Dry run to see what would be deleted
        themis clean --dry-run

        # Remove runs older than 30 days
        themis clean --older-than 30
    """
    logging_utils.configure_logging(level="info")

    from themis.storage import ExperimentStorage

    storage_root = _resolve_storage_root(storage)
    if not storage_root.exists():
        print(f"No storage found at {storage_root}")
        return 1

    if older_than is None:
        print("Error: --older-than is required to clean runs")
        return 1

    storage_backend = ExperimentStorage(storage_root)
    runs = storage_backend.list_runs()
    cutoff = datetime.now() - timedelta(days=older_than)

    candidates = []
    for run in runs:
        try:
            created_at = datetime.fromisoformat(run.created_at)
        except ValueError:
            continue
        if created_at < cutoff:
            candidates.append(run)

    if not candidates:
        print("No runs matched the cleanup criteria.")
        return 0

    if dry_run:
        print("Runs to delete:")
        for run in candidates:
            print(f"  - {run.run_id} (created {run.created_at})")
        return 0

    deleted = 0
    for run in candidates:
        storage_backend.delete_run(run.run_id)
        deleted += 1

    print(f"Deleted {deleted} run(s).")
    return 0


def _resolve_storage_root(storage: str | None) -> Path:
    if storage:
        return Path(storage).expanduser()
    env_storage = os.getenv("THEMIS_STORAGE")
    if env_storage:
        return Path(env_storage).expanduser()
    return Path(".cache/experiments")


_PROMPT_FIELD_CANDIDATES = ("prompt", "question", "input", "text", "query", "problem")
_REFERENCE_FIELD_CANDIDATES = (
    "answer",
    "reference",
    "expected",
    "label",
    "target",
    "solution",
)
_ID_FIELD_CANDIDATES = ("id", "sample_id", "dataset_id", "unique_id", "uid")


def _load_custom_dataset_file(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    rows = _read_dataset_rows(path)
    if not rows:
        raise ValueError(f"Dataset file is empty: {path}")

    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Row {index} in {path} must be a JSON object, got {type(row).__name__}."
            )
        normalized_rows.append(dict(row))

    prompt_field = _detect_required_field(
        normalized_rows, _PROMPT_FIELD_CANDIDATES, "prompt"
    )
    reference_field = _detect_required_field(
        normalized_rows, _REFERENCE_FIELD_CANDIDATES, "reference"
    )
    id_field = _detect_optional_field(normalized_rows, _ID_FIELD_CANDIDATES)

    for index, row in enumerate(normalized_rows, 1):
        if id_field is None:
            row["id"] = str(index)
        elif id_field != "id":
            row["id"] = row[id_field]

        if reference_field not in ("answer", "reference"):
            row.setdefault("reference", row[reference_field])

    return normalized_rows, prompt_field, reference_field


def _read_dataset_rows(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                content = line.strip()
                if not content:
                    continue
                try:
                    rows.append(json.loads(content))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} in {path}: {exc.msg}"
                    ) from exc
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
        if not isinstance(payload, builtins.list):
            raise ValueError(
                f"JSON dataset in {path} must be a top-level array of row objects."
            )
        return payload
    raise ValueError(
        f"Unsupported dataset format '{suffix or '<none>'}' for {path}. "
        "Use .json or .jsonl."
    )


def _detect_required_field(
    rows: list[dict[str, Any]], candidates: tuple[str, ...], kind: str
) -> str:
    field = _detect_optional_field(rows, candidates)
    if field is None:
        options = ", ".join(candidates)
        raise ValueError(
            f"Could not detect {kind} field in dataset. Add one of: {options}."
        )
    return field


def _detect_optional_field(
    rows: list[dict[str, Any]], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if all(candidate in row for row in rows):
            return candidate
    return None


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
    parsed_argv = builtins.list(argv) if argv is not None else None
    try:
        result = app(parsed_argv)
    except SystemExit as exc:  # pragma: no cover - CLI integration path
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import logging

        # If logger is configured, this will show up. If not, it might be lost if no handler.
        # But we configure logging in commands. If it fails before command execution (e.g. parsing),
        # we might not have logging configured.
        # Let's try to get logger anyway.
        logger = logging.getLogger(__name__)
        logger.error("Unexpected error", exc_info=True)
        return 1
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
