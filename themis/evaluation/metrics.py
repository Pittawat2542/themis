"""Evaluation metric primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.evaluation import math_verify_utils
from themis.interfaces import Metric as MetricInterface


def _normalize_text(value: str, case_sensitive: bool, strip_whitespace: bool) -> str:
    if strip_whitespace:
        value = value.strip()
    if not case_sensitive:
        value = value.lower()
    return value


@dataclass
class ExactMatch(MetricInterface):
    case_sensitive: bool = False
    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        self.name = "ExactMatch"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        metadata = dict(metadata or {})
        normalized_prediction = _normalize_text(
            str(prediction), self.case_sensitive, self.strip_whitespace
        )
        matched_reference: str | None = None
        for reference in references:
            normalized_reference = _normalize_text(
                str(reference), self.case_sensitive, self.strip_whitespace
            )
            if normalized_prediction == normalized_reference:
                matched_reference = str(reference)
                break
        value = 1.0 if matched_reference is not None else 0.0
        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details={"matched_reference": matched_reference},
            metadata=metadata,
        )


@dataclass
class LengthDifferenceTolerance(MetricInterface):
    max_delta: int = 0

    def __post_init__(self) -> None:
        self.name = "LengthDifferenceTolerance"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        metadata = dict(metadata or {})
        reference = str(references[0]) if references else ""
        delta = abs(len(str(prediction)) - len(reference))
        value = 1.0 if delta <= self.max_delta else 0.0
        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details={"delta": delta},
            metadata=metadata,
        )


@dataclass
class CompositeMetric(MetricInterface):
    children: Sequence[MetricInterface]

    def __post_init__(self) -> None:
        self.name = "CompositeMetric"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        child_results = [
            child.compute(
                prediction=prediction, references=references, metadata=metadata
            )
            for child in self.children
        ]
        if not child_results:
            return core_entities.MetricScore(
                metric_name=self.name,
                value=0.0,
                details={},
                metadata=dict(metadata or {}),
            )
        value = sum(result.value for result in child_results) / len(child_results)
        details = {result.metric_name: result.details for result in child_results}
        return core_entities.MetricScore(
            metric_name=self.name,
            value=value,
            details=details,
            metadata=dict(metadata or {}),
        )


@dataclass
class ResponseLength(MetricInterface):
    """Reports the length of the prediction response."""

    def __post_init__(self) -> None:
        self.name = "ResponseLength"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        metadata = dict(metadata or {})
        text = str(prediction)
        length = len(text)
        return core_entities.MetricScore(
            metric_name=self.name,
            value=float(length),
            details={"length": length},
            metadata=metadata,
        )


@dataclass
class MathVerifyAccuracy(MetricInterface):
    """Numeric equivalence check using math-verify."""

    def __post_init__(self) -> None:
        math_verify_utils.require_math_verify()
        self.name = "MathVerifyAccuracy"

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        math_verify_utils.require_math_verify()
        metadata = dict(metadata or {})
        prediction_expr = math_verify_utils.parse_expression(str(prediction))
        passed = False
        for reference in references:
            reference_expr = math_verify_utils.parse_expression(str(reference))
            if math_verify_utils.verify_expressions(reference_expr, prediction_expr):
                passed = True
                break
        return core_entities.MetricScore(
            metric_name=self.name,
            value=1.0 if passed else 0.0,
            details={"verified": passed},
            metadata=metadata,
        )
