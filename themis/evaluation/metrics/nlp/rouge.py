"""ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric.

ROUGE measures overlap between generated text and reference summaries
using n-grams and longest common subsequence.

References:
    Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


class ROUGEVariant(str, Enum):
    """ROUGE metric variants."""
    
    ROUGE_1 = "rouge1"  # Unigram overlap
    ROUGE_2 = "rouge2"  # Bigram overlap
    ROUGE_L = "rougeL"  # Longest common subsequence
    ROUGE_L_SUM = "rougeLsum"  # LCS with summary-level computation


class ROUGE(Metric):
    """ROUGE metric using rouge-score library.
    
    ROUGE is a recall-oriented metric that measures n-gram overlap between
    generated text and reference summaries. It's commonly used for evaluating
    text summarization and text generation tasks.
    
    Variants:
        - ROUGE-1: Unigram overlap
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence (sentence-level)
        - ROUGE-Lsum: Longest common subsequence (summary-level)
    
    Attributes:
        name: Metric identifier (e.g., "rouge1", "rouge2", "rougeL")
        variant: Which ROUGE variant to compute
        use_stemmer: Whether to use Porter stemmer
    
    Example:
        >>> from themis.evaluation.metrics.nlp import ROUGE, ROUGEVariant
        >>> metric = ROUGE(variant=ROUGEVariant.ROUGE_2)
        >>> score = metric.compute(
        ...     prediction="The quick brown fox jumps over the lazy dog",
        ...     references=["A quick brown fox jumped over a lazy dog"]
        ... )
        >>> print(f"ROUGE-2 F1: {score.value:.4f}")
        ROUGE-2 F1: 0.6154
    """
    
    requires_reference = True
    
    def __init__(
        self,
        variant: ROUGEVariant = ROUGEVariant.ROUGE_L,
        use_stemmer: bool = True,
    ):
        """Initialize ROUGE metric.
        
        Args:
            variant: Which ROUGE variant to compute
            use_stemmer: Whether to use Porter stemmer for word matching
        """
        self.variant = variant
        self.use_stemmer = use_stemmer
        self.name = variant.value
        
        # Lazy import rouge-score (not required for all users)
        try:
            from rouge_score import rouge_scorer
            self._scorer = rouge_scorer.RougeScorer(
                [variant.value],
                use_stemmer=use_stemmer,
            )
        except ImportError:
            raise ImportError(
                "rouge-score is required for ROUGE metric. "
                "Install it with: pip install rouge-score"
            )
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute ROUGE score.
        
        Args:
            prediction: Generated text (already extracted by pipeline)
            references: List of reference summaries
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with ROUGE F1 score and precision/recall details
        """
        # Convert to strings
        pred_str = str(prediction)
        ref_strs = [str(ref) for ref in references]
        
        # Compute ROUGE for each reference and take the maximum
        max_precision = 0.0
        max_recall = 0.0
        max_f1 = 0.0
        
        for ref_str in ref_strs:
            scores = self._scorer.score(ref_str, pred_str)
            rouge_score = scores[self.variant.value]
            
            if rouge_score.fmeasure > max_f1:
                max_precision = rouge_score.precision
                max_recall = rouge_score.recall
                max_f1 = rouge_score.fmeasure
        
        return MetricScore(
            metric_name=self.name,
            value=max_f1,  # Use F1 as primary score
            details={
                "precision": max_precision,
                "recall": max_recall,
                "f1": max_f1,
                "variant": self.variant.value,
                "num_references": len(ref_strs),
            },
            metadata=metadata or {},
        )


__all__ = ["ROUGE", "ROUGEVariant"]
