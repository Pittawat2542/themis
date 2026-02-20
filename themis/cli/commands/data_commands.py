"""Data management commands."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.cli.utils import resolve_storage_root
from themis.utils import logging_utils


def list_command(
    what: Annotated[str, Parameter(name="WHAT", show_default=False)],
    *,
    storage: Annotated[str | None, Parameter(help="Storage path for runs")] = None,
    limit: Annotated[int | None, Parameter(help="Limit number of results")] = None,
    verbose: Annotated[bool, Parameter(help="Show detailed information")] = False,
) -> int:
    """List runs, benchmarks, or available metrics."""
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

        storage_root = resolve_storage_root(storage)
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


def clean_command(
    *,
    storage: Annotated[str | None, Parameter(help="Storage path to clean")] = None,
    older_than: Annotated[
        int | None, Parameter(help="Remove runs older than N days")
    ] = None,
    dry_run: Annotated[bool, Parameter(help="Show what would be deleted")] = False,
) -> int:
    """Clean up old runs and cached data."""
    logging_utils.configure_logging(level="info")

    from themis.storage import ExperimentStorage

    storage_root = resolve_storage_root(storage)
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


def share_command(
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
    """Generate a shareable results badge + Markdown snippet for a run."""
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
