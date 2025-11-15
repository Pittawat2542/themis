"""CSV export functionality for experiment reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import MutableMapping, Sequence

from themis.core import entities as core_entities
from themis.experiment import orchestrator


def export_report_csv(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    include_failures: bool = True,
) -> Path:
    """Write per-sample metrics to a CSV file for offline analysis.

    Args:
        report: Experiment report to export
        path: Output path for CSV file
        include_failures: Whether to include failures column

    Returns:
        Path to created CSV file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_condition, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )

    # Create a proper index mapping generation records to their metadata
    gen_record_index = {}
    for gen_record in report.generation_results:
        sample_id = gen_record.task.metadata.get(
            "dataset_id"
        ) or gen_record.task.metadata.get("sample_id")
        prompt_template = gen_record.task.prompt.spec.name
        model_identifier = gen_record.task.model.identifier
        sampling_temp = gen_record.task.sampling.temperature
        sampling_max_tokens = gen_record.task.sampling.max_tokens
        condition_id = f"{sample_id}_{prompt_template}_{model_identifier}_{sampling_temp}_{sampling_max_tokens}"
        gen_record_index[condition_id] = gen_record

    metric_names = sorted(report.evaluation_report.metrics.keys())
    fieldnames = (
        ["sample_id"] + metadata_fields + [f"metric:{name}" for name in metric_names]
    )
    if include_failures:
        fieldnames.append("failures")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        # Process evaluation records in the same order as generation records
        for i, eval_record in enumerate(report.evaluation_report.records):
            # Find the corresponding generation record by index
            if i < len(report.generation_results):
                gen_record = report.generation_results[i]
                sample_id = gen_record.task.metadata.get(
                    "dataset_id"
                ) or gen_record.task.metadata.get("sample_id")
                prompt_template = gen_record.task.prompt.spec.name
                model_identifier = gen_record.task.model.identifier
                sampling_temp = gen_record.task.sampling.temperature
                sampling_max_tokens = gen_record.task.sampling.max_tokens
                condition_id = f"{sample_id}_{prompt_template}_{model_identifier}_{sampling_temp}_{sampling_max_tokens}"
                metadata = metadata_by_condition.get(condition_id, {})
            else:
                # Fallback for extra evaluation records
                sample_id = eval_record.sample_id or ""
                metadata = {}

            row: dict[str, object] = {"sample_id": sample_id}
            for field in metadata_fields:
                row[field] = metadata.get(field, "")
            score_by_name = {
                score.metric_name: score.value for score in eval_record.scores
            }
            for name in metric_names:
                row[f"metric:{name}"] = score_by_name.get(name, "")
            if include_failures:
                row["failures"] = "; ".join(eval_record.failures)
            writer.writerow(row)
    return path


def _collect_sample_metadata(
    records: Sequence[core_entities.GenerationRecord],
) -> tuple[dict[str, MutableMapping[str, object]], list[str]]:
    """Collect metadata from generation records.

    Args:
        records: Generation records

    Returns:
        Tuple of (metadata by condition ID, list of metadata fields)
    """
    metadata: dict[str, MutableMapping[str, object]] = {}
    for index, record in enumerate(records):
        sample_id = _extract_sample_id(record.task.metadata)
        if sample_id is None:
            sample_id = f"sample-{index}"

        # Create unique identifier for each experimental condition
        prompt_template = record.task.prompt.spec.name
        model_identifier = record.task.model.identifier
        sampling_temp = record.task.sampling.temperature
        sampling_max_tokens = record.task.sampling.max_tokens

        # Create unique condition key
        condition_id = f"{sample_id}_{prompt_template}_{model_identifier}_{sampling_temp}_{sampling_max_tokens}"

        # Store metadata with unique condition ID
        condition_metadata = _metadata_from_task(record)
        metadata[condition_id] = condition_metadata

    # Collect all field names from all conditions
    fields = sorted({field for meta in metadata.values() for field in meta.keys()})

    return metadata, fields


def _extract_sample_id(metadata: dict[str, object]) -> str | None:
    """Extract sample ID from metadata.

    Args:
        metadata: Task metadata

    Returns:
        Sample ID or None
    """
    value = metadata.get("dataset_id") or metadata.get("sample_id")
    if value is None:
        return None
    return str(value)


def _metadata_from_task(record: core_entities.GenerationRecord) -> dict[str, object]:
    """Build metadata dict from generation record.

    Args:
        record: Generation record

    Returns:
        Metadata dictionary
    """
    metadata = dict(record.task.metadata)
    metadata.setdefault("model_identifier", record.task.model.identifier)
    metadata.setdefault("model_provider", record.task.model.provider)
    metadata.setdefault("prompt_template", record.task.prompt.spec.name)
    metadata.setdefault("sampling_temperature", record.task.sampling.temperature)
    metadata.setdefault("sampling_top_p", record.task.sampling.top_p)
    metadata.setdefault("sampling_max_tokens", record.task.sampling.max_tokens)
    return metadata
