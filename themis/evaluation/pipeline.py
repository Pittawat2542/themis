"""Evaluation pipeline orchestration.

This module provides two complementary pipeline styles:

1. EvaluationPipeline: Traditional batch evaluation with extractors, metrics, and strategies
2. ComposableEvaluationPipeline: Chainable builder pattern for composing evaluation steps

Example (Traditional):
    >>> pipeline = EvaluationPipeline(
    ...     extractor=JsonFieldExtractor("answer"),
    ...     metrics=[ExactMatch()]
    ... )
    >>> report = pipeline.evaluate(records)

Example (Composable):
    >>> pipeline = (
    ...     ComposableEvaluationPipeline()
    ...     .extract(JsonFieldExtractor("answer"))
    ...     .validate(lambda x: isinstance(x, str), "Must be string")
    ...     .transform(lambda x: x.strip().lower(), name="normalize")
    ...     .compute_metrics([ExactMatch()], references=["42"])
    ... )
    >>> result = pipeline.evaluate(record)
"""

from __future__ import annotations

# Re-export pipeline implementations for backward compatibility
from themis.evaluation.pipelines.composable_pipeline import (
    ComposableEvaluationPipeline,
    EvaluationResult,
    EvaluationStep,
)
from themis.evaluation.pipelines.standard_pipeline import EvaluationPipeline
from themis.evaluation.reports import (
    EvaluationFailure,
    EvaluationReport,
    MetricAggregate,
)

__all__ = [
    "EvaluationPipeline",
    "ComposableEvaluationPipeline",
    "EvaluationStep",
    "EvaluationResult",
    "MetricAggregate",
    "EvaluationReport",
    "EvaluationFailure",
]
