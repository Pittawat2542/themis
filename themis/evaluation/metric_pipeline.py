"""Primary metric evaluation pipeline for vNext workflows."""

from __future__ import annotations

from themis.evaluation.pipelines.standard_pipeline import EvaluationPipeline


class MetricPipeline(EvaluationPipeline):
    """Primary evaluation pipeline for vNext (alias of standard pipeline)."""


__all__ = ["MetricPipeline"]
