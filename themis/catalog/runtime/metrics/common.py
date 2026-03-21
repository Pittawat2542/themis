"""Catalog runtime shared metrics."""

from __future__ import annotations

from themis.extractors.builtin import _normalize_text
from themis.records import MetricScore

from ..common import _coerce_float, _expected_text


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference is not None else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == _expected_text(context)),
        )


class NormalizedExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        return MetricScore(
            metric_id="normalized_exact_match",
            value=float(str(parsed) == _normalize_text(_expected_text(context))),
        )


class ChoiceAccuracyMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        expected = _expected_text(context).strip().upper()
        return MetricScore(
            metric_id="choice_accuracy",
            value=float(str(parsed).strip().upper() == expected),
        )


class NumericExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else None
        )
        actual = _coerce_float(parsed)
        expected = _coerce_float(_expected_text(context))
        return MetricScore(
            metric_id="numeric_exact_match",
            value=float(
                actual is not None and expected is not None and actual == expected
            ),
        )
