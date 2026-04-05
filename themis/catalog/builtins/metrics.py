"""Builtin pure metrics."""

from __future__ import annotations

import importlib
from collections import Counter
from typing import Any

from themis.core.base import JSONValue
from themis.core.contexts import ScoreContext
from themis.core.models import Case, ParsedOutput, Score, ScoreError


class ExactMatchMetric:
    component_id = "builtin/exact_match"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-exact-match-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        return Score(
            metric_id=self.component_id,
            value=float(parsed.value == case.expected_output),
        )


class F1Metric:
    component_id = "builtin/f1"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-f1-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        predicted = _tokenize(parsed.value)
        expected = _tokenize(case.expected_output)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if not predicted and not expected:
            value = 1.0
        elif overlap == 0:
            value = 0.0
        else:
            precision = overlap / len(predicted)
            recall = overlap / len(expected)
            value = (2 * precision * recall) / (precision + recall)
        return Score(metric_id=self.component_id, value=value)


class BleuMetric:
    component_id = "builtin/bleu"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-bleu-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        predicted = _tokenize(parsed.value)
        expected = Counter(_tokenize(case.expected_output))
        if not predicted:
            value = 0.0
        else:
            matches = 0
            remaining = expected.copy()
            for token in predicted:
                if remaining[token] > 0:
                    matches += 1
                    remaining[token] -= 1
            value = matches / len(predicted)
        return Score(metric_id=self.component_id, value=value)


class ChoiceAccuracyMetric:
    component_id = "builtin/choice_accuracy"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-choice-accuracy-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        expected = _expected_text(case.expected_output, key="choice").strip().upper()
        actual = str(parsed.value).strip().upper()
        return Score(metric_id=self.component_id, value=float(actual == expected))


class MathEquivalenceMetric:
    component_id = "builtin/math_equivalence"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-math-equivalence-fingerprint"

    def score(
        self, parsed: ParsedOutput, case: Case, ctx: ScoreContext
    ) -> Score | ScoreError:
        del ctx
        gold_answer = _expected_text(case.expected_output, key="answer").strip()
        candidate_answer = str(parsed.value).strip()
        details: dict[str, JSONValue] = {
            "candidate_answer": candidate_answer,
            "gold_answer": gold_answer,
        }
        if not candidate_answer or not gold_answer:
            return Score(metric_id=self.component_id, value=0.0, details=details)
        try:
            math_verify = _import_math_verify()
        except RuntimeError:
            normalized_candidate = _normalize_math_text(candidate_answer)
            normalized_gold = _normalize_math_text(gold_answer)
            return Score(
                metric_id=self.component_id,
                value=float(normalized_candidate == normalized_gold),
                details={
                    **details,
                    "fallback": "normalized_text",
                },
            )
        try:
            parsed_gold = math_verify.parse(gold_answer)
            parsed_candidate = math_verify.parse(candidate_answer)
            verified = math_verify.verify(parsed_gold, parsed_candidate)
        except Exception as exc:
            return ScoreError(
                metric_id=self.component_id,
                reason=str(exc),
                details=details,
            )
        equivalent = bool(verified[0] if isinstance(verified, tuple) else verified)
        return Score(
            metric_id=self.component_id,
            value=float(equivalent),
            details=details,
        )


class ProcbenchFinalAccuracyMetric:
    component_id = "builtin/procbench_final_accuracy"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-procbench-final-accuracy-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        actual = _canonical_procbench_value(parsed.value)
        expected = _canonical_procbench_value(
            _expected_value(case.expected_output, key="answer")
        )
        return Score(
            metric_id=self.component_id,
            value=float(actual == expected),
            details=_json_details(
                {
                    "canonical_actual": actual,
                    "canonical_expected": expected,
                }
            ),
        )


def _tokenize(value: object) -> list[str]:
    if value is None:
        return []
    return str(value).lower().split()


def _expected_value(value: object, *, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return value


def _expected_text(value: object, *, key: str) -> str:
    resolved = _expected_value(value, key=key)
    return "" if resolved is None else str(resolved)


def _import_math_verify():
    try:
        return importlib.import_module("math_verify")
    except ImportError as exc:
        raise RuntimeError(
            'Math equivalence requires the optional math dependency. Install it with: uv add "themis-eval[math]"'
        ) from exc


def _canonical_procbench_value(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        return stripped
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
    return str(value)


def _normalize_math_text(value: str) -> str:
    return value.replace(" ", "").strip().lower()


def _json_details(value: dict[str, object]) -> dict[str, JSONValue]:
    return {
        key: item
        if isinstance(item, (str, int, float, bool)) or item is None
        else str(item)
        for key, item in value.items()
    }
