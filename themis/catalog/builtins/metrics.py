"""Builtin pure metrics."""

from __future__ import annotations

from collections import Counter

from themis.core.contexts import ScoreContext
from themis.core.models import Case, ParsedOutput, Score


class ExactMatchMetric:
    component_id = "builtin/exact_match"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "builtin-exact-match-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        return Score(metric_id=self.component_id, value=float(parsed.value == case.expected_output))


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


def _tokenize(value: object) -> list[str]:
    if value is None:
        return []
    return str(value).lower().split()
