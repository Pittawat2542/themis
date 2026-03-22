"""Procbench-specific final-answer metric."""

from __future__ import annotations

import json

from themis.extractors.builtin import extract_embedded_json_payload
from themis.records import MetricScore


class ProcbenchFinalAccuracyMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference is not None else ""
        expected = context.get("expected")
        canonical_actual = _canonical_procbench_value(actual)
        canonical_expected = _canonical_procbench_value(expected)
        return MetricScore(
            metric_id="procbench_final_accuracy",
            value=float(canonical_actual == canonical_expected),
            details={
                "canonical_actual": canonical_actual,
                "canonical_expected": canonical_expected,
            },
        )


def _canonical_procbench_value(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = extract_embedded_json_payload(stripped)
        except Exception:
            return stripped
        return _canonical_procbench_value(parsed)
    if isinstance(value, list):
        return [_canonical_procbench_value(item) for item in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if value is None:
        return None
    return json.loads(json.dumps(value, sort_keys=True))
