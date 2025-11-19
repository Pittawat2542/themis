"""Multi-experiment comparison commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.experiment.comparison import compare_experiments, diff_configs


def compare_command(
    *,
    run_ids: Annotated[
        list[str],
        Parameter(
            help="Run IDs to compare (comma-separated or multiple --run-ids)",
        ),
    ],
    storage: Annotated[
        Path,
        Parameter(
            help="Storage directory containing experiment results",
        ),
    ] = Path(".cache/runs"),
    metrics: Annotated[
        list[str] | None,
        Parameter(
            help="Metrics to compare (default: all available)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            help="Output file path (format inferred from extension: .csv, .md, .json)",
        ),
    ] = None,
    format: Annotated[
        str,
        Parameter(
            help="Output format: csv, markdown, json, latex",
        ),
    ] = "markdown",
    highlight_best: Annotated[
        str | None,
        Parameter(
            help="Metric to highlight best performer (e.g., 'accuracy')",
        ),
    ] = None,
) -> int:
    """Compare multiple experiment runs.

    Automatically includes cost data when available. Costs are tracked
    automatically during experiment runs and displayed in comparisons.

    Examples:
        # Compare three runs with default metrics (includes cost if tracked)
        uv run python -m themis.cli compare \\
          --run-ids run-1 run-2 run-3 \\
          --storage .cache/runs

        # Compare with specific metrics, export to CSV
        uv run python -m themis.cli compare \\
          --run-ids run-1 run-2 run-3 \\
          --metrics accuracy \\
          --output comparison.csv

        # Use 'cost' as a metric for ranking and Pareto analysis
        uv run python -m themis.cli pareto \\
          --run-ids run-1 run-2 run-3 \\
          --objectives accuracy cost \\
          --maximize true false

        # Highlight best accuracy performer
        uv run python -m themis.cli compare \\
          --run-ids run-1 run-2 run-3 \\
          --highlight-best accuracy
    """
    try:
        # Load and compare experiments
        print(f"Loading experiments from {storage}...")
        comparison = compare_experiments(
            run_ids=run_ids,
            storage_dir=storage,
            metrics=metrics,
            include_metadata=True,
        )

        print(f"\n✓ Loaded {len(comparison.experiments)} experiments")
        print(f"  Metrics: {', '.join(comparison.metrics)}\n")

        # Display comparison table
        print("=" * 80)
        print("Experiment Comparison")
        print("=" * 80)

        # Check if any experiment has cost data
        has_cost = any(exp.get_cost() is not None for exp in comparison.experiments)

        # Header
        header_cols = ["Run ID"] + comparison.metrics + ["Samples", "Failures"]
        if has_cost:
            header_cols.append("Cost ($)")
        col_widths = [max(20, len(col)) for col in header_cols]

        header = " | ".join(
            col.ljust(width) for col, width in zip(header_cols, col_widths)
        )
        print(header)
        print("-" * len(header))

        # Rows
        for exp in comparison.experiments:
            row_values = [exp.run_id[:20]]  # Truncate run ID
            for metric in comparison.metrics:
                val = exp.get_metric(metric)
                row_values.append(f"{val:.4f}" if val is not None else "N/A")
            row_values.append(str(exp.sample_count))
            row_values.append(str(exp.failure_count))

            # Add cost if available
            if has_cost:
                cost = exp.get_cost()
                row_values.append(f"{cost:.4f}" if cost is not None else "N/A")

            row = " | ".join(
                val.ljust(width) for val, width in zip(row_values, col_widths)
            )
            print(row)

        print("=" * 80)

        # Highlight best if requested
        if highlight_best:
            if highlight_best in comparison.metrics:
                best = comparison.highlight_best(highlight_best)
                if best:
                    best_value = best.get_metric(highlight_best)
                    print(
                        f"\n⭐ Best {highlight_best}: {best.run_id} ({best_value:.4f})"
                    )
                else:
                    print(f"\n⚠️  No valid values for metric '{highlight_best}'")
            else:
                print(
                    f"\n⚠️  Metric '{highlight_best}' not found. Available: {comparison.metrics}"
                )

        # Export if requested
        if output:
            output = Path(output)
            # Infer format from extension if not specified
            if output.suffix == ".csv":
                comparison.to_csv(output)
                print(f"\n✓ Exported to {output} (CSV)")
            elif output.suffix == ".md":
                comparison.to_markdown(output)
                print(f"\n✓ Exported to {output} (Markdown)")
            elif output.suffix == ".json":
                import json

                output.write_text(
                    json.dumps(comparison.to_dict(), indent=2), encoding="utf-8"
                )
                print(f"\n✓ Exported to {output} (JSON)")
            elif output.suffix == ".tex":
                comparison.to_latex(output, style="booktabs")
                print(f"\n✓ Exported to {output} (LaTeX)")
            else:
                # Use specified format
                if format == "csv":
                    comparison.to_csv(output)
                    print(f"\n✓ Exported to {output} (CSV)")
                elif format == "markdown":
                    comparison.to_markdown(output)
                    print(f"\n✓ Exported to {output} (Markdown)")
                elif format == "json":
                    import json

                    output.write_text(
                        json.dumps(comparison.to_dict(), indent=2), encoding="utf-8"
                    )
                    print(f"\n✓ Exported to {output} (JSON)")
                elif format == "latex":
                    comparison.to_latex(output, style="booktabs")
                    print(f"\n✓ Exported to {output} (LaTeX)")
                else:
                    print(f"\n⚠️  Unknown format: {format}")
                    print("Available formats: csv, markdown, json, latex")
                    return 1

        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def diff_command(
    *,
    run_id_a: Annotated[
        str,
        Parameter(
            help="First run ID",
        ),
    ],
    run_id_b: Annotated[
        str,
        Parameter(
            help="Second run ID",
        ),
    ],
    storage: Annotated[
        Path,
        Parameter(
            help="Storage directory containing experiment results",
        ),
    ] = Path(".cache/runs"),
) -> int:
    """Show configuration differences between two experiment runs.

    Examples:
        # Compare configurations
        uv run python -m themis.cli diff \\
          --run-id-a run-1 \\
          --run-id-b run-2 \\
          --storage .cache/runs
    """
    try:
        diff = diff_configs(run_id_a, run_id_b, storage)

        print("=" * 80)
        print(f"Configuration Diff: {run_id_a} → {run_id_b}")
        print("=" * 80)

        if not diff.has_differences():
            print("\n✓ No differences found - configurations are identical\n")
            return 0

        # Show changed fields
        if diff.changed_fields:
            print("\n📝 Changed Fields:")
            for key, (old, new) in diff.changed_fields.items():
                print(f"\n  {key}:")
                print(f"    - {run_id_a}: {old}")
                print(f"    + {run_id_b}: {new}")

        # Show added fields
        if diff.added_fields:
            print("\n➕ Added Fields (in run_id_b):")
            for key, value in diff.added_fields.items():
                print(f"  {key}: {value}")

        # Show removed fields
        if diff.removed_fields:
            print("\n➖ Removed Fields (from run_id_a):")
            for key, value in diff.removed_fields.items():
                print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure both run IDs exist and have config.json files.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def pareto_command(
    *,
    run_ids: Annotated[
        list[str],
        Parameter(
            help="Run IDs to analyze",
        ),
    ],
    storage: Annotated[
        Path,
        Parameter(
            help="Storage directory containing experiment results",
        ),
    ] = Path(".cache/runs"),
    objectives: Annotated[
        list[str],
        Parameter(
            help="Metrics to optimize (e.g., accuracy cost)",
        ),
    ],
    maximize: Annotated[
        list[bool] | None,
        Parameter(
            help="Whether to maximize each objective (true/false for each)",
        ),
    ] = None,
) -> int:
    """Find Pareto-optimal experiments across multiple objectives.

    The Pareto frontier consists of experiments where no other experiment
    is better on all objectives simultaneously.

    Examples:
        # Find experiments with best accuracy/cost tradeoff
        # (maximize accuracy, minimize cost)
        uv run python -m themis.cli pareto \\
          --run-ids run-1 run-2 run-3 run-4 \\
          --objectives accuracy cost \\
          --maximize true false

        # Find experiments with best accuracy/latency tradeoff
        uv run python -m themis.cli pareto \\
          --run-ids run-1 run-2 run-3 \\
          --objectives accuracy latency \\
          --maximize true false
    """
    try:
        # Load experiments
        print(f"Loading experiments from {storage}...")
        comparison = compare_experiments(
            run_ids=run_ids,
            storage_dir=storage,
            metrics=objectives,
            include_metadata=True,
        )

        print(f"\n✓ Loaded {len(comparison.experiments)} experiments")
        print(f"  Objectives: {', '.join(objectives)}\n")

        # Compute Pareto frontier
        pareto_ids = comparison.pareto_frontier(objectives, maximize)

        print("=" * 80)
        print("Pareto Frontier Analysis")
        print("=" * 80)

        if not pareto_ids:
            print(
                "\n⚠️  No Pareto-optimal experiments found (all experiments have missing values)\n"
            )
            return 0

        print(f"\n⭐ Found {len(pareto_ids)} Pareto-optimal experiment(s):\n")

        # Show Pareto-optimal experiments
        for run_id in pareto_ids:
            exp = next(e for e in comparison.experiments if e.run_id == run_id)
            print(f"  • {run_id}")
            for obj in objectives:
                val = exp.get_metric(obj)
                print(
                    f"      {obj}: {val:.4f}"
                    if val is not None
                    else f"      {obj}: N/A"
                )

        # Show dominated experiments
        dominated = [
            exp for exp in comparison.experiments if exp.run_id not in pareto_ids
        ]
        if dominated:
            print(f"\n📊 Dominated experiments ({len(dominated)}):")
            for exp in dominated:
                print(f"  • {exp.run_id}")

        print("\n" + "=" * 80)
        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


__all__ = ["compare_command", "diff_command", "pareto_command"]
