"""Private helpers shared across catalog common modules."""

from __future__ import annotations

import math

from themis import BenchmarkDefinition
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict


def _catalog_metadata(definition: BenchmarkDefinition) -> JSONDict:
    return validate_json_dict(
        definition.metadata,
        label=f"{definition.family} benchmark metadata",
    )


def _catalog_metadata_str(definition: BenchmarkDefinition, key: str) -> str:
    value = _catalog_metadata(definition).get(key)
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        f"Built-in benchmark '{definition.benchmark_id}' metadata must define '{key}'."
    )


def _catalog_metadata_optional_str(
    definition: BenchmarkDefinition, key: str
) -> str | None:
    value = _catalog_metadata(definition).get(key)
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        f"Built-in benchmark '{definition.benchmark_id}' metadata '{key}' must be a non-empty string when present."
    )


def _primary_metric_id(definition: BenchmarkDefinition) -> str:
    if definition.primary_metric_id:
        return definition.primary_metric_id
    raise ValueError(
        f"Built-in benchmark '{definition.benchmark_id}' does not define a primary metric."
    )


def _hle_variant_ids(definition: BenchmarkDefinition) -> list[str]:
    metadata = _catalog_metadata(definition)
    value = metadata.get("variant_ids")
    if isinstance(value, list):
        variant_ids = [str(item) for item in value if str(item)]
        if variant_ids:
            return variant_ids
    raise ValueError(
        "Built-in benchmark 'hle' requires explicit HLE variants in the benchmark id."
    )


def _hle_prompt_template(variant_id: str) -> str:
    preamble = ""
    if variant_id == "no_tool":
        preamble = "Do not use tools. Answer directly from the provided context.\n\n"
    return (
        f"{preamble}"
        "Your response should be in the following format:\n"
        "Explanation: your explanation for your answer choice\n"
        "Answer: your chosen answer\n"
        "Confidence: your confidence score between 0% and 100% "
        "for your answer\n\nQuestion:\n{question}"
    )


def _hle_calibration_error(
    *,
    confidences: list[float],
    truths: list[float],
    accuracy: float,
) -> float:
    if not confidences:
        return 0.0
    if len(confidences) < 100:
        incorrect_deltas = [
            abs(confidence - accuracy)
            for confidence, truth in zip(confidences, truths, strict=True)
            if truth < 1.0
        ]
        if incorrect_deltas:
            return sum(incorrect_deltas) / len(incorrect_deltas)
        return abs(sum(confidences) / len(confidences) - accuracy)
    paired = sorted(zip(confidences, truths, strict=True), key=lambda pair: pair[0])
    beta = 100
    calibration = 0.0
    total_examples = len(paired)
    for start in range(0, total_examples, beta):
        bucket = paired[start : start + beta]
        if not bucket:
            continue
        bucket_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        bucket_accuracy = sum(truth for _, truth in bucket) / len(bucket)
        difference = abs(bucket_confidence - bucket_accuracy)
        calibration += len(bucket) / total_examples * (difference**2)
    return math.sqrt(calibration)


def _estimate_pass_at_k_payload(
    totals: list[int],
    correct_counts: list[int],
) -> JSONDict:
    if not totals or not correct_counts:
        return _json_dict({}, label="pass@k payload")
    payload: dict[str, float] = {}
    for k in (1, 10, 100):
        if min(totals) < k:
            continue
        values = [
            _estimate_pass_at_k(n, c, k)
            for n, c in zip(totals, correct_counts, strict=True)
        ]
        payload[f"pass@{k}"] = sum(values) / len(values)
    return _json_dict(payload, label="pass@k payload")


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def _json_dict(value: object, *, label: str) -> JSONDict:
    return validate_json_dict(value, label=label)


def _detail_mapping(value: object) -> JSONDict:
    if not isinstance(value, dict):
        return {}
    return validate_json_dict(
        {str(key): item for key, item in value.items()},
        label="detail mapping",
    )


def _detail_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _coerce_score_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_score_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
