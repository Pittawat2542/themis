from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from themis.core import entities as core_entities
from themis.evaluation import math_verify_utils
from themis.interfaces import Metric as MetricInterface


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
