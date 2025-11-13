from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
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
