"""JSON export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

from themis.core import entities as core_entities
from themis.experiment import orchestrator
from themis.experiment.export._shared import (
    _collect_sample_metadata,
    _extract_sample_id,
    _chart_to_dict,
    ChartLike,
)


def export_report_json(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int | None = None,
    indent: int = 2,
) -> Path:
    """Serialize the report details to JSON for downstream tooling."""

    payload = build_json_report(
        report,
        charts=charts,
        title=title,
        sample_limit=sample_limit,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def export_summary_json(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    run_id: str | None = None,
    indent: int = 2,
) -> Path:
    """Export a lightweight summary JSON file for quick results viewing.

    This creates a small summary file (~1KB) containing only the essential
    metrics and metadata, without the full sample-level details. This is
    ideal for quickly comparing multiple runs without parsing large report files.

    Args:
        report: Experiment report to summarize
        path: Output path for summary.json
        run_id: Optional run identifier to include in summary
        indent: JSON indentation level

    Returns:
        Path to the created summary file

    Example:
        >>> export_summary_json(report, "outputs/run-123/summary.json", run_id="run-123")
        >>> # Quick comparison: cat outputs/*/summary.json | jq '.accuracy'

    Note:
        The summary file is typically ~1KB compared to ~1.6MB for the full report.
        This makes it 1000x faster to view and compare results across runs.
    """
    # Extract key metrics
    metrics_summary = {}
    for name, aggregate in report.evaluation_report.metrics.items():
        metrics_summary[name] = {
            "mean": aggregate.mean,
            "count": aggregate.count,
        }

    # Extract metadata from first generation record
    metadata = {}
    if report.generation_results:
        first_record = report.generation_results[0]
        metadata = {
            "model": first_record.task.model.identifier,
            "prompt_template": first_record.task.prompt.spec.name,
            "sampling": {
                "temperature": first_record.task.sampling.temperature,
                "top_p": first_record.task.sampling.top_p,
                "max_tokens": first_record.task.sampling.max_tokens,
            },
        }

    # Calculate total cost if available
    total_cost = 0.0
    for record in report.generation_results:
        if "cost_usd" in record.metrics:
            total_cost += record.metrics["cost_usd"]

    # Count failures
    failure_count = len(report.evaluation_report.failures)

    # Build summary
    summary = {
        "run_id": run_id,
        "total_samples": len(report.generation_results),
        "metrics": metrics_summary,
        "metadata": metadata,
        "cost_usd": round(total_cost, 4) if total_cost > 0 else None,
        "failures": failure_count,
        "failure_rate": (
            round(failure_count / len(report.generation_results), 4)
            if report.generation_results
            else 0.0
        ),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=indent), encoding="utf-8")
    return path


def build_json_report(
    report: orchestrator.ExperimentReport,
    *,
    charts: Sequence[ChartLike] | None = None,
    title: str = "Experiment report",
    sample_limit: int | None = None,
) -> dict[str, object]:
    metadata_by_sample, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )
    metric_names = sorted(report.evaluation_report.metrics.keys())
    samples = []
    limit = (
        sample_limit
        if sample_limit is not None
        else len(report.evaluation_report.records)
    )

    # Build mapping from sample_id to generation records to get task info
    gen_records_by_sample: dict[str, core_entities.GenerationRecord] = {}
    for gen_record in report.generation_results:
        sid = _extract_sample_id(gen_record.task.metadata)
        if sid:
            # Use first generation record for each sample (may have multiple with different conditions)
            if sid not in gen_records_by_sample:
                gen_records_by_sample[sid] = gen_record

    for index, record in enumerate(report.evaluation_report.records):
        if index >= limit:
            break
        sample_id = record.sample_id or ""

        # Try to find corresponding generation record for this evaluation record
        gen_record = gen_records_by_sample.get(sample_id)

        # Build condition_id if we have the generation record
        sample_metadata = {}
        if gen_record is not None:
            prompt_template = gen_record.task.prompt.spec.name
            model_identifier = gen_record.task.model.identifier
            sampling_temp = gen_record.task.sampling.temperature
            sampling_max_tokens = gen_record.task.sampling.max_tokens
            condition_id = f"{sample_id}_{prompt_template}_{model_identifier}_{sampling_temp}_{sampling_max_tokens}"
            sample_metadata = dict(metadata_by_sample.get(condition_id, {}))

        scores = [
            {
                "metric": score.metric_name,
                "value": score.value,
                "details": score.details,
                "metadata": score.metadata,
            }
            for score in record.scores
        ]
        samples.append(
            {
                "sample_id": sample_id,
                "metadata": sample_metadata,
                "scores": scores,
                "failures": list(record.failures),
            }
        )

    payload = {
        "title": title,
        "summary": {
            **report.metadata,
            "run_failures": len(report.failures),
            "evaluation_failures": len(report.evaluation_report.failures),
        },
        "metrics": [
            {
                "name": name,
                "count": metric.count,
                "mean": metric.mean,
            }
            for name, metric in sorted(
                report.evaluation_report.metrics.items(), key=lambda item: item[0]
            )
        ],
        "samples": samples,
        "rendered_sample_limit": limit,
        "total_samples": len(report.evaluation_report.records),
        "charts": [
            chart.as_dict() if hasattr(chart, "as_dict") else _chart_to_dict(chart)
            for chart in charts or ()
        ],
        "run_failures": [
            {"sample_id": failure.sample_id, "message": failure.message}
            for failure in report.failures
        ],
        "evaluation_failures": [
            {"sample_id": failure.sample_id, "message": failure.message}
            for failure in report.evaluation_report.failures
        ],
        "metrics_rendered": metric_names,
    }
    return payload
