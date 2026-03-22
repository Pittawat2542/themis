"""Catalog runtime shared metrics."""

from __future__ import annotations

from themis._optional import import_optional
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


class MathEquivalenceMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else None
        )
        raw_text = ""
        inference = getattr(candidate, "inference", None)
        if inference is not None and getattr(inference, "raw_text", None) is not None:
            raw_text = str(inference.raw_text)
        candidate_answer = str(parsed if parsed is not None else raw_text).strip()
        gold_answer = _expected_text(context).strip()
        details = {
            "candidate_answer": candidate_answer,
            "gold_answer": gold_answer,
            "used_boxed_answer": "\\boxed" in raw_text,
        }
        if not candidate_answer or not gold_answer:
            return MetricScore(
                metric_id="math_equivalence",
                value=0.0,
                details=details,
            )
        try:
            math_verify = import_optional("math_verify", extra="math")
        except Exception as exc:
            return MetricScore(
                metric_id="math_equivalence",
                value=0.0,
                details=details,
                error=str(exc),
            )
        try:
            parsed_gold = math_verify.parse(gold_answer)
            parsed_candidate = math_verify.parse(candidate_answer)
            verified = math_verify.verify(parsed_gold, parsed_candidate)
            equivalent = bool(verified[0] if isinstance(verified, tuple) else verified)
            return MetricScore(
                metric_id="math_equivalence",
                value=float(equivalent),
                details=details,
            )
        except Exception as exc:
            details["verification_error"] = str(exc)
            return MetricScore(
                metric_id="math_equivalence",
                value=0.0,
                details=details,
                error=str(exc),
            )
