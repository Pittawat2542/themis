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

# vNext: protocol definition for evaluation pipelines
from typing import Protocol, Sequence, runtime_checkable

# Re-export pipeline implementations for backward compatibility
from themis.evaluation.pipelines.composable_pipeline import (
    ComposableEvaluationPipeline,
    ComposableEvaluationReportPipeline,
    EvaluationResult,
    EvaluationStep,
)
from themis.evaluation.pipelines.standard_pipeline import EvaluationPipeline
from themis.evaluation.reports import (
    EvaluationFailure,
    EvaluationReport,
    MetricAggregate,
)
from themis.core import entities as core_entities


@runtime_checkable
class EvaluationPipelineContract(Protocol):
    """Contract for evaluation pipelines."""

    def evaluate(
        self, records: Sequence[core_entities.GenerationRecord]
    ) -> EvaluationReport:  # pragma: no cover - protocol
        ...

    def evaluation_fingerprint(self) -> dict:  # pragma: no cover - protocol
        ...

__all__ = [
    "EvaluationPipeline",
    "EvaluationPipelineContract",
    "ComposableEvaluationPipeline",
    "ComposableEvaluationReportPipeline",
    "EvaluationStep",
    "EvaluationResult",
    "MetricAggregate",
    "EvaluationReport",
    "EvaluationFailure",
]
