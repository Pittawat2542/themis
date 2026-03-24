"""Catalog result summarizers and score helpers."""

from __future__ import annotations

import math

from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict

from ._shared import (
    _coerce_score_float,
    _coerce_score_int,
    _detail_mapping,
    _detail_str_list,
    _estimate_pass_at_k_payload,
    _hle_calibration_error,
    _hle_variant_ids,
    _json_dict,
    _primary_metric_id,
)


def iter_score_rows(result, metric_id: str) -> list[ScoreRow]:
    return list(
        result.projection_repo.iter_candidate_scores(
            trial_hashes=result.trial_hashes,
            metric_id=metric_id,
            evaluation_hash=getattr(result, "active_evaluation_hash", None),
        )
    )


def mean_summary(metric_id: str, result) -> JSONDict:
    rows = iter_score_rows(result, metric_id)
    count = len(rows)
    mean = sum(row.score for row in rows) / count if count else 0.0
    return _json_dict(
        {"metric_id": metric_id, "count": count, "mean": mean},
        label=f"{metric_id} summary",
    )


def summarize_simpleqa(_definition, result) -> JSONDict:
    rows = iter_score_rows(result, "simpleqa_verified_score")
    count = len(rows)
    if count == 0:
        return _json_dict(
            {
                "metric_id": "simpleqa_verified_score",
                "count": 0,
                "correct_rate": 0.0,
                "attempted_rate": 0.0,
                "accuracy_given_attempted": 0.0,
                "f1": 0.0,
            },
            label="simpleqa summary",
        )
    correct_rate = (
        sum(1.0 for row in rows if row.details.get("grade") == "CORRECT") / count
    )
    incorrect_rate = (
        sum(1.0 for row in rows if row.details.get("grade") == "INCORRECT") / count
    )
    attempted_rate = correct_rate + incorrect_rate
    accuracy_given_attempted = (
        correct_rate / attempted_rate if attempted_rate > 0 else 0.0
    )
    f1 = (
        2
        * accuracy_given_attempted
        * correct_rate
        / (accuracy_given_attempted + correct_rate)
        if (accuracy_given_attempted + correct_rate) > 0
        else 0.0
    )
    return _json_dict(
        {
            "metric_id": "simpleqa_verified_score",
            "count": count,
            "correct_rate": correct_rate,
            "attempted_rate": attempted_rate,
            "accuracy_given_attempted": accuracy_given_attempted,
            "f1": f1,
        },
        label="simpleqa summary",
    )


def summarize_healthbench(_definition, result) -> JSONDict:
    rows = iter_score_rows(result, "healthbench_score")
    count = len(rows)
    mean_score = sum(row.score for row in rows) / count if count else 0.0
    tag_values: dict[str, list[float]] = {}
    for row in rows:
        for tag in _detail_str_list(row.details.get("example_tags")):
            if isinstance(tag, str):
                tag_values.setdefault(tag, []).append(row.score)
    return _json_dict(
        {
            "metric_id": "healthbench_score",
            "count": count,
            "mean_overall_score": mean_score,
            "tag_means": {
                tag: sum(values) / len(values)
                for tag, values in sorted(tag_values.items())
            },
        },
        label="healthbench summary",
    )


def summarize_hle(definition, result) -> JSONDict:
    variant_ids = _hle_variant_ids(definition)
    if len(variant_ids) == 1:
        summary = _summarize_hle_variant(
            iter_score_rows(result, "hle_accuracy"),
            scan_stats=_detail_mapping(getattr(result, "scan_stats", {}) or {}),
        )
        return _json_dict(
            {
                "metric_id": "hle_accuracy",
                **summary,
            },
            label="hle summary",
        )
    rows = iter_score_rows(result, "hle_accuracy")
    summaries_by_trial_hash = {}
    if hasattr(result, "iter_trial_summaries"):
        summaries_by_trial_hash = {
            row.trial_hash: row
            for row in result.iter_trial_summaries()  # type: ignore[attr-defined]
        }
    grouped_rows: dict[str, list[ScoreRow]] = {
        variant_id: [] for variant_id in variant_ids
    }
    for row in rows:
        summary_row = summaries_by_trial_hash.get(row.trial_hash)
        variant_id = None
        if summary_row is not None and hasattr(summary_row, "dimensions"):
            dimensions = getattr(summary_row, "dimensions", {}) or {}
            if isinstance(dimensions, dict):
                candidate = dimensions.get("hle_variant")
                if isinstance(candidate, str):
                    variant_id = candidate
        if variant_id is None and hasattr(summary_row, "slice_id"):
            slice_id = getattr(summary_row, "slice_id", None)
            if isinstance(slice_id, str) and slice_id.startswith("hle-"):
                variant_id = slice_id.removeprefix("hle-")
        if variant_id is None:
            prompt_variant_id = getattr(row, "prompt_variant_id", None)
            if isinstance(prompt_variant_id, str) and prompt_variant_id.startswith(
                "hle-"
            ):
                variant_id = prompt_variant_id.removeprefix("hle-").removesuffix(
                    "-default"
                )
        if variant_id in grouped_rows:
            grouped_rows[variant_id].append(row)
    scan_stats = _detail_mapping(getattr(result, "scan_stats", {}) or {})
    return _json_dict(
        {
            "metric_id": "hle_accuracy",
            "variant_ids": variant_ids,
            "variants": {
                variant_id: _summarize_hle_variant(
                    grouped_rows[variant_id],
                    scan_stats=_detail_mapping(scan_stats.get(variant_id, {}) or {}),
                )
                for variant_id in variant_ids
            },
        },
        label="hle summary",
    )


def summarize_mcq(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_math(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_lpfqa(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_codeforces(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_aethercode(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_livecodebench(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_humaneval(definition, result) -> JSONDict:
    return _summarize_humaneval_pass_at_k(
        result,
        metric_id=(
            _primary_metric_id(definition)
            if definition is not None
            else "humaneval_pass_rate"
        ),
        include_plus=False,
    )


def summarize_humaneval_plus(definition, result) -> JSONDict:
    return _summarize_humaneval_pass_at_k(
        result,
        metric_id=(
            _primary_metric_id(definition)
            if definition is not None
            else "humaneval_plus_pass_rate"
        ),
        include_plus=True,
    )


def _summarize_hle_variant(
    rows: list[ScoreRow],
    *,
    scan_stats: JSONDict,
) -> JSONDict:
    count = len(rows)
    accuracy = sum(row.score for row in rows) / count if count else 0.0
    confidence_interval_half_width = (
        1.96 * math.sqrt(accuracy * (1 - accuracy) / count) if count else 0.0
    )
    confidences = [
        max(
            0.0,
            min(1.0, _coerce_score_float(row.details.get("confidence"), 100.0) / 100.0),
        )
        for row in rows
    ]
    truths = [float(bool(row.details.get("correct", False))) for row in rows]
    calibration_error = _hle_calibration_error(
        confidences=confidences,
        truths=truths,
        accuracy=accuracy,
    )
    return _json_dict(
        {
            "count": count,
            "accuracy": accuracy,
            "confidence_interval_half_width": confidence_interval_half_width,
            "calibration_error": calibration_error,
            "skipped_image_count": _coerce_score_int(
                scan_stats.get("skipped_image_count"),
                0,
            ),
        },
        label="hle summary",
    )


def _summarize_humaneval_pass_at_k(
    result,
    *,
    metric_id: str,
    include_plus: bool,
) -> JSONDict:
    summaries = {
        row.trial_hash: row for row in getattr(result, "iter_trial_summaries")()
    }
    grouped: dict[str, list[ScoreRow]] = {}
    for row in iter_score_rows(result, metric_id):
        summary = summaries.get(row.trial_hash)
        if summary is None or summary.item_id is None:
            continue
        grouped.setdefault(summary.item_id, []).append(row)

    totals = [len(rows) for rows in grouped.values() if rows]
    sample_count_min = min(totals) if totals else 0
    base_correct = [
        sum(1 for row in rows if row.details.get("base_status") == "pass")
        for rows in grouped.values()
    ]
    payload: JSONDict = {
        "metric_id": metric_id,
        "task_count": len(grouped),
        "sample_count_min": sample_count_min,
        "base_pass_at_k": _estimate_pass_at_k_payload(totals, base_correct),
    }
    if include_plus:
        plus_correct = [
            sum(
                1
                for row in rows
                if row.details.get("base_status") == "pass"
                and row.details.get("plus_status") == "pass"
            )
            for rows in grouped.values()
        ]
        payload["plus_pass_at_k"] = _estimate_pass_at_k_payload(totals, plus_correct)
    return _json_dict(payload, label="humaneval summary")
