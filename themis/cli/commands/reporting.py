"""Reporting CLI commands."""

from __future__ import annotations

from typing import Literal

from themis.cli.helpers import (
    build_run_query,
    initialize_store,
    load_experiment,
    resolve_persisted_run_id,
)
from themis.core.reporter import Reporter


def report(
    *,
    config: str,
    run_id: str | None = None,
    baseline_label: str | None = None,
    dataset_source_id: str | None = None,
    dataset_fingerprint: str | None = None,
    metric_id: str | None = None,
    tag: list[str] | None = None,
    lineage_parent_run_id: str | None = None,
    status: str | None = None,
    format: Literal["json", "markdown", "csv", "latex"] = "json",
) -> int:
    experiment = load_experiment(config)
    store = initialize_store(experiment)
    query = build_run_query(
        dataset_source_id=dataset_source_id,
        dataset_fingerprint=dataset_fingerprint,
        metric_id=metric_id,
        tags=tag,
        lineage_parent_run_id=lineage_parent_run_id,
        status=status,
    )
    resolved_run_id = (
        resolve_persisted_run_id(
            store,
            run_id=run_id,
            baseline_label=baseline_label,
            query=query,
        )
        if any(
            (
                run_id is not None,
                baseline_label is not None,
                dataset_source_id is not None,
                dataset_fingerprint is not None,
                metric_id is not None,
                bool(tag),
                lineage_parent_run_id is not None,
                status is not None,
            )
        )
        else experiment.compile().run_id
    )
    reporter = Reporter(store)
    if format == "json":
        print(reporter.export_json(resolved_run_id))
        return 0
    if format == "markdown":
        print(reporter.export_markdown(resolved_run_id))
        return 0
    if format == "csv":
        print(reporter.export_csv(resolved_run_id))
        return 0
    print(reporter.export_latex(resolved_run_id))
    return 0
