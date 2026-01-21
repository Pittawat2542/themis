"""Quick results viewing commands for experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter


def summary_command(
    *,
    run_id: Annotated[
        str,
        Parameter(
            help="Run ID to view summary for",
        ),
    ],
    storage: Annotated[
        Path,
        Parameter(
            help="Storage directory containing experiment results",
        ),
    ] = Path(".cache/runs"),
) -> int:
    """View quick summary of a single experiment run.

    This command reads the lightweight summary.json file (~1KB) instead of
    the full report.json (~1.6MB), making it much faster for quick checks.

    Examples:
        # View summary for a specific run
        uv run python -m themis.cli results summary \\
          --run-id run-20260118-032014 \\
          --storage outputs/evaluation

        # Quick check of latest run
        uv run python -m themis.cli results summary \\
          --run-id $(ls -t outputs/evaluation | head -1)
    """
    try:
        # Try to find summary.json
        run_dir = storage / run_id
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            print(f"Error: Summary file not found at {summary_path}")
            print("\nNote: summary.json is only available for runs created with")
            print("the updated export functionality. For older runs, use the")
            print("'compare' command which reads full report.json files.")
            return 1

        # Load summary
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        # Display summary
        print("=" * 80)
        print(f"Experiment Summary: {run_id}")
        print("=" * 80)

        # Basic info
        print(f"\nRun ID: {summary.get('run_id', 'N/A')}")
        print(f"Total Samples: {summary.get('total_samples', 0)}")

        # Metadata
        metadata = summary.get("metadata", {})
        if metadata:
            print("\nConfiguration:")
            print(f"  Model: {metadata.get('model', 'N/A')}")
            print(f"  Prompt: {metadata.get('prompt_template', 'N/A')}")
            sampling = metadata.get("sampling", {})
            if sampling:
                print(f"  Temperature: {sampling.get('temperature', 'N/A')}")
                print(f"  Max Tokens: {sampling.get('max_tokens', 'N/A')}")

        # Metrics
        metrics = summary.get("metrics", {})
        if metrics:
            print("\nMetrics:")
            for name, data in metrics.items():
                mean = data.get("mean", 0)
                count = data.get("count", 0)
                print(f"  {name}: {mean:.4f} (n={count})")

        # Cost
        cost = summary.get("cost_usd")
        if cost is not None:
            print(f"\nCost: ${cost:.4f}")

        # Failures
        failures = summary.get("failures", 0)
        failure_rate = summary.get("failure_rate", 0)
        if failures > 0:
            print(f"\nFailures: {failures} ({failure_rate:.2%})")

        print("\n" + "=" * 80)
        return 0

    except FileNotFoundError:
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def list_command(
    *,
    storage: Annotated[
        Path,
        Parameter(
            help="Storage directory containing experiment results",
        ),
    ] = Path(".cache/runs"),
    limit: Annotated[
        int | None,
        Parameter(
            help="Maximum number of runs to display",
        ),
    ] = None,
    sort_by: Annotated[
        str,
        Parameter(
            help="Sort runs by: time (newest first) or metric name",
        ),
    ] = "time",
) -> int:
    """List all experiment runs with quick summaries.

    This command scans for summary.json files and displays a table of all runs.
    Much faster than loading full report.json files.

    Examples:
        # List all runs
        uv run python -m themis.cli results list

        # List 10 most recent runs
        uv run python -m themis.cli results list --limit 10

        # List runs sorted by accuracy
        uv run python -m themis.cli results list --sort-by accuracy
    """
    try:
        if not storage.exists():
            print(f"Error: Storage directory not found: {storage}")
            return 1

        # Find all summary.json files
        summaries = []
        for run_dir in storage.iterdir():
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                try:
                    with summary_path.open("r", encoding="utf-8") as f:
                        summary = json.load(f)
                    summary["_run_dir"] = run_dir.name
                    summary["_mtime"] = summary_path.stat().st_mtime
                    summaries.append(summary)
                except Exception:
                    continue

        if not summaries:
            print(f"No experiment runs found in {storage}")
            print("\nNote: Only runs with summary.json files are shown.")
            return 0

        # Sort summaries
        if sort_by == "time":
            summaries.sort(key=lambda s: s.get("_mtime", 0), reverse=True)
        else:
            # Sort by metric value
            summaries.sort(
                key=lambda s: s.get("metrics", {}).get(sort_by, {}).get("mean", 0),
                reverse=True,
            )

        # Apply limit
        if limit:
            summaries = summaries[:limit]

        # Display table
        print("=" * 120)
        print(f"Found {len(summaries)} experiment run(s)")
        print("=" * 120)

        # Collect all metric names
        all_metrics = set()
        for s in summaries:
            all_metrics.update(s.get("metrics", {}).keys())
        metric_names = sorted(all_metrics)

        # Header
        header_cols = ["Run ID", "Model", "Samples"] + metric_names + ["Cost ($)"]
        col_widths = [25, 30, 8] + [12] * len(metric_names) + [10]

        header = " | ".join(
            col.ljust(width)[:width] for col, width in zip(header_cols, col_widths)
        )
        print(header)
        print("-" * len(header))

        # Rows
        for summary in summaries:
            run_id = summary.get("_run_dir", "N/A")[:25]
            model = summary.get("metadata", {}).get("model", "N/A")[:30]
            samples = str(summary.get("total_samples", 0))
            cost = summary.get("cost_usd")

            row_values = [run_id, model, samples]

            # Add metric values
            for metric_name in metric_names:
                metric_data = summary.get("metrics", {}).get(metric_name, {})
                mean = metric_data.get("mean")
                if mean is not None:
                    row_values.append(f"{mean:.4f}")
                else:
                    row_values.append("N/A")

            # Add cost
            if cost is not None:
                row_values.append(f"{cost:.4f}")
            else:
                row_values.append("N/A")

            row = " | ".join(
                val.ljust(width)[:width] for val, width in zip(row_values, col_widths)
            )
            print(row)

        print("=" * 120)
        return 0

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


__all__ = ["summary_command", "list_command"]
