"""CSV export utilities."""

from __future__ import annotations

import csv
from pathlib import Path

from themis.experiment import orchestrator
from themis.experiment.export._shared import _collect_sample_metadata


def export_report_csv(
    report: orchestrator.ExperimentReport,
    path: str | Path,
    *,
    include_failures: bool = True,
) -> Path:
    """Write per-sample metrics to a CSV file for offline analysis."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_condition, metadata_fields = _collect_sample_metadata(
        report.generation_results
    )

    # Create a proper index mapping generation records to their metadata
    # We assume evaluation records are in the same order as generation records
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
