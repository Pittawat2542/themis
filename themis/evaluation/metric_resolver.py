"""Metric resolution logic."""

from __future__ import annotations

import logging
import re
from typing import Any

from themis.evaluation.metrics.exact_match import ExactMatch
from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy
from themis.evaluation.metrics.response_length import ResponseLength
from themis.exceptions import MetricError

logger = logging.getLogger(__name__)

# Module-level metrics registry for custom metrics
_METRICS_REGISTRY: dict[str, type] = {}


def register_metric(name: str, metric_cls: type) -> None:
    """Register a custom metric for use in evaluate().

    This allows users to add their own metrics to Themis without modifying
    the source code. Registered metrics can be used by passing their names
    to the `metrics` parameter in evaluate().

    Args:
        name: Metric name (used in evaluate(metrics=[name]))
        metric_cls: Metric class implementing the Metric interface.
            Must have a compute() method that takes prediction, references,
            and metadata parameters.

    Raises:
        MetricError: If metric_cls is not a class or doesn't implement compute()

    Example:
        >>> from themis.evaluation.metrics import MyCustomMetric
        >>> themis.register_metric("my_metric", MyCustomMetric)
        >>> report = themis.evaluate("math500", model="gpt-4", metrics=["my_metric"])
    """
    if not isinstance(metric_cls, type):
        raise MetricError(f"metric_cls must be a class, got {type(metric_cls)}")

    # Validate that it implements the Metric interface
    if not hasattr(metric_cls, "compute"):
        raise MetricError(
            f"{metric_cls.__name__} must implement compute() method. "
            f"See themis.evaluation.metrics for examples."
        )

    _METRICS_REGISTRY[name] = metric_cls
    logger.info(f"Registered custom metric: {name} -> {metric_cls.__name__}")


def get_registered_metrics() -> dict[str, type]:
    """Get all currently registered custom metrics.

    Returns:
        Dictionary mapping metric names to their classes
    """
    return _METRICS_REGISTRY.copy()


_BUILTIN_METRICS: dict[str, Any] = {
    "exact_match": ExactMatch,
    "math_verify": MathVerifyAccuracy,
    "response_length": ResponseLength,
}

# NLP metrics (Phase 2)
try:
    from themis.evaluation.metrics.nlp import (
        BLEU,
        ROUGE,
        BERTScore,
        METEOR,
        ROUGEVariant,
    )

    _BUILTIN_METRICS.update(
        {
            "bleu": BLEU,
            "rouge1": lambda: ROUGE(variant=ROUGEVariant.ROUGE_1),
            "rouge2": lambda: ROUGE(variant=ROUGEVariant.ROUGE_2),
            "rougeL": lambda: ROUGE(variant=ROUGEVariant.ROUGE_L),
            "bertscore": BERTScore,
            "meteor": METEOR,
        }
    )
except ImportError:
    pass

# Code metrics (some optional dependencies)
try:
    from themis.evaluation.metrics.code.execution import ExecutionAccuracy
    from themis.evaluation.metrics.code.pass_at_k import PassAtK

    _BUILTIN_METRICS["pass_at_k"] = PassAtK
    _BUILTIN_METRICS["execution_accuracy"] = ExecutionAccuracy

    try:
        from themis.evaluation.metrics.code.codebleu import CodeBLEU

        _BUILTIN_METRICS["codebleu"] = CodeBLEU
    except ImportError:
        pass
except ImportError:
    pass


def resolve_metrics(metric_names: list[str]) -> list[Any]:
    """Resolve metric names to metric instances.

    Args:
        metric_names: List of metric names (e.g., ["exact_match", "bleu"])

    Returns:
        List of metric instances

    Raises:
        MetricError: If a metric name is unknown
    """
    # Merge built-in and custom metrics
    # Custom metrics can override built-in metrics
    registry = {**_BUILTIN_METRICS, **_METRICS_REGISTRY}

    def _normalize_metric_name(name: str) -> str | None:
        raw = name.strip()
        if raw in registry:
            return raw
        lowered = raw.lower()
        if lowered in registry:
            return lowered
        for key in registry:
            if key.lower() == lowered:
                return key
        # Convert CamelCase / PascalCase to snake_case
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
        if snake in registry:
            return snake
        return None

    metrics = []
    for name in metric_names:
        resolved = _normalize_metric_name(name)
        if resolved is None:
            available = ", ".join(sorted(registry.keys()))
            raise MetricError(f"Unknown metric: {name}. Available metrics: {available}")

        metric_cls = registry[resolved]
        metrics.append(metric_cls())

    return metrics


def _reset_metrics_for_testing() -> None:
    """Clear custom metrics registry. For testing only."""
    _METRICS_REGISTRY.clear()
