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
class ConsistencyMetric(MetricInterface):
    case_sensitive: bool = False
    strip_whitespace: bool = True

    def __post_init__(self) -> None:
        self.name = "Consistency"
        self.requires_reference = False

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> core_entities.MetricScore:
        md = dict(metadata or {})

        outputs: list[str]
        if isinstance(prediction, (list, tuple)):
            outputs = [str(p) for p in prediction]
        else:
            outputs = [str(prediction)]

        normalized = [
            _normalize_text(text, self.case_sensitive, self.strip_whitespace)
            for text in outputs
        ]

        majority_correct = None
        reference_text = None
        if references:
            reference_text = _normalize_text(
                str(references[0]), self.case_sensitive, self.strip_whitespace
            )
            correct = [1.0 if out == reference_text else 0.0 for out in normalized]
            majority_correct = sum(correct) / max(1, len(correct))

        from collections import Counter

        counter = Counter(normalized)
        mode_count = max(counter.values()) if counter else 0
        agreement = mode_count / max(1, len(normalized))

        flips = 0
        for i in range(1, len(normalized)):
            if normalized[i] != normalized[i - 1]:
                flips += 1
        flip_rate = flips / max(1, len(normalized) - 1)

        value = majority_correct if majority_correct is not None else agreement

        return core_entities.MetricScore(
            metric_name=self.name,
            value=float(value),
            details={
                "agreement": agreement,
                "flip_rate": flip_rate,
                "outputs": outputs,
                "reference": reference_text,
            },
            metadata=md,
        )
