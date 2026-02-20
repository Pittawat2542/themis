"""Evaluation commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.cli.utils import (
    load_custom_dataset_file,
    resolve_storage_root,
)
from themis.utils import logging_utils


def eval_command(
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
    """Run an evaluation on a benchmark or custom dataset."""
    from themis.experiment import export as experiment_export

    # Configure logging
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
        storage_root = resolve_storage_root(storage)

        from themis.api import evaluate as evaluate_api

        if is_custom_dataset:
            print(f"Loading custom dataset from: {custom_dataset_path}")
            dataset_rows, detected_prompt_field, _ = load_custom_dataset_file(
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
            report = evaluate_api(
                benchmark_or_dataset,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                workers=workers,
                storage=storage_root,
                run_id=run_id,
                resume=resume,
                limit=limit,
            )

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        # Print metrics
        eval_report = report.evaluation_report
        if eval_report and eval_report.metrics:
            print("\nMetrics:")
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


def demo_command(
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
    return eval_command(
        "demo",
        model=model,
        limit=limit,
        verbose=verbose,
        json_logs=json_logs,
    )
