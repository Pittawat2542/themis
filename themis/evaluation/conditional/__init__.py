"""Conditional and adaptive evaluation strategies.

This module provides evaluation components that adapt based on sample characteristics:
- ConditionalMetric: Only runs when condition is met
- AdaptiveEvaluationPipeline: Selects metrics based on sample metadata
- Metric selectors: Helper functions for common selection patterns
"""

from themis.evaluation.conditional.metric import ConditionalMetric
from themis.evaluation.conditional.pipeline import AdaptiveEvaluationPipeline
from themis.evaluation.conditional.selectors import (
    select_by_metadata_field,
    select_by_difficulty,
    select_by_condition,
    combine_selectors,
)

__all__ = [
    "ConditionalMetric",
    "AdaptiveEvaluationPipeline",
    "select_by_metadata_field",
    "select_by_difficulty",
    "select_by_condition",
    "combine_selectors",
]
