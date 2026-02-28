"""Shared types and helpers for export module."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Protocol

from themis.core import entities as core_entities


class ChartPointLike(Protocol):
    label: str
    x_value: object
    metric_value: float
    metric_name: str
    count: int


class ChartLike(Protocol):
    title: str
    x_label: str
    y_label: str
    metric_name: str
    points: Sequence[ChartPointLike]


def _collect_sample_metadata(
    records: Sequence[core_entities.GenerationRecord],
) -> tuple[dict[str, MutableMapping[str, object]], list[str]]:
    metadata: dict[str, MutableMapping[str, object]] = {}
    for index, record in enumerate(records):
        sample_id = _extract_sample_id(record.task.metadata)
        if sample_id is None:
            sample_id = f"sample-{index}"

        # Create unique identifier for each experimental condition
        # Include prompt template, model, and sampling to distinguish conditions
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


def _extract_sample_id(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("dataset_id") or metadata.get("sample_id")
    if value is None:
        return None
    return str(value)


def _metadata_from_task(record: core_entities.GenerationRecord) -> dict[str, object]:
    metadata = dict(record.task.metadata)
    metadata.setdefault("model_identifier", record.task.model.identifier)
    metadata.setdefault("model_provider", record.task.model.provider)
    metadata.setdefault("prompt_template", record.task.prompt.spec.name)
    metadata.setdefault("sampling_temperature", record.task.sampling.temperature)
    metadata.setdefault("sampling_top_p", record.task.sampling.top_p)
    metadata.setdefault("sampling_max_tokens", record.task.sampling.max_tokens)
    return metadata


def _chart_to_dict(chart: ChartLike) -> dict[str, object]:
    return {
        "title": chart.title,
        "x_label": chart.x_label,
        "y_label": chart.y_label,
        "metric": chart.metric_name,
        "points": [
            {
                "label": point.label,
                "x": getattr(point, "x_value", getattr(point, "x", None)),
                "value": point.metric_value,
                "count": point.count,
            }
            for point in chart.points
        ],
    }
