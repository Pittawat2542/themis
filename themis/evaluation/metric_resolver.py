"""Metric resolution logic."""

from __future__ import annotations

import logging
import re
from typing import Any

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
        TypeError: If metric_cls is not a class
        ValueError: If metric_cls doesn't implement the required interface

    Example:
        >>> from themis.evaluation.metrics import MyCustomMetric
        >>> themis.register_metric("my_metric", MyCustomMetric)
        >>> report = themis.evaluate("math500", model="gpt-4", metrics=["my_metric"])
    """
    if not isinstance(metric_cls, type):
        raise TypeError(f"metric_cls must be a class, got {type(metric_cls)}")

    # Validate that it implements the Metric interface
    if not hasattr(metric_cls, "compute"):
        raise ValueError(
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


def resolve_metrics(metric_names: list[str]) -> list:
    """Resolve metric names to metric instances.

    Args:
        metric_names: List of metric names (e.g., ["exact_match", "bleu"])

    Returns:
        List of metric instances

    Raises:
        ValueError: If a metric name is unknown
    """
    from themis.evaluation.metrics.exact_match import ExactMatch
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy
    from themis.evaluation.metrics.response_length import ResponseLength

    # NLP metrics (Phase 2)
    try:
        from themis.evaluation.metrics.nlp import (
            BLEU,
            ROUGE,
            BERTScore,
            METEOR,
            ROUGEVariant,
        )

        nlp_available = True
    except ImportError:
        nlp_available = False

    # Code metrics (some optional dependencies)
    try:
        from themis.evaluation.metrics.code.execution import ExecutionAccuracy
        from themis.evaluation.metrics.code.pass_at_k import PassAtK

        code_metrics: dict[str, Any] = {
            "pass_at_k": PassAtK,
            "execution_accuracy": ExecutionAccuracy,
        }
        try:
            from themis.evaluation.metrics.code.codebleu import CodeBLEU

            code_metrics["codebleu"] = CodeBLEU
        except ImportError:
            pass
    except ImportError:
        code_metrics = {}

    BUILTIN_METRICS: dict[str, Any] = {
        # Core metrics
        "exact_match": ExactMatch,
        "math_verify": MathVerifyAccuracy,
        "response_length": ResponseLength,
    }

    # Add NLP metrics if available
    if nlp_available:
        BUILTIN_METRICS.update(
            {
                "bleu": BLEU,
                "rouge1": lambda: ROUGE(variant=ROUGEVariant.ROUGE_1),
                "rouge2": lambda: ROUGE(variant=ROUGEVariant.ROUGE_2),
                "rougeL": lambda: ROUGE(variant=ROUGEVariant.ROUGE_L),
                "bertscore": BERTScore,
                "meteor": METEOR,
            }  # type: ignore
        )

    BUILTIN_METRICS.update(code_metrics)

    # Merge built-in and custom metrics
    # Custom metrics can override built-in metrics
    METRICS_REGISTRY = {**BUILTIN_METRICS, **_METRICS_REGISTRY}

    def _normalize_metric_name(name: str) -> str | None:
        raw = name.strip()
        if raw in METRICS_REGISTRY:
            return raw
        lowered = raw.lower()
        if lowered in METRICS_REGISTRY:
            return lowered
        for key in METRICS_REGISTRY.keys():
            if key.lower() == lowered:
                return key
        # Convert CamelCase / PascalCase to snake_case
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
        if snake in METRICS_REGISTRY:
            return snake
        return None

    metrics = []
    for name in metric_names:
        resolved = _normalize_metric_name(name)
        if resolved is None:
            available = ", ".join(sorted(METRICS_REGISTRY.keys()))
            raise ValueError(f"Unknown metric: {name}. Available metrics: {available}")

        metric_cls = METRICS_REGISTRY[resolved]
        # Handle both class and lambda factory
        if callable(metric_cls) and not isinstance(metric_cls, type):
            metrics.append(metric_cls())
        else:
            metrics.append(metric_cls())

    return metrics
