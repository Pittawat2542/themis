"""BERTScore metric implementation.

BERTScore computes similarity using contextual embeddings from BERT-like models
instead of exact word matches.

References:
    Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT.
"""

from __future__ import annotations

from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


class BERTScore(Metric):
    """BERTScore metric using bert-score library.
    
    BERTScore leverages contextual embeddings from pre-trained models (BERT, RoBERTa, etc.)
    to compute semantic similarity between generated and reference texts. It's more
    robust to paraphrasing than exact n-gram matching methods.
    
    The metric computes token-level cosine similarity between embeddings and aggregates
    using precision, recall, and F1.
    
    Attributes:
        name: Metric identifier ("bertscore")
        model_type: Pre-trained model to use for embeddings
        lang: Language code for automatic model selection
        rescale_with_baseline: Whether to rescale scores using baseline
    
    Example:
        >>> from themis.evaluation.metrics.nlp import BERTScore
        >>> metric = BERTScore(model_type="microsoft/deberta-xlarge-mnli")
        >>> score = metric.compute(
        ...     prediction="The cat sat on the mat",
        ...     references=["A cat is sitting on a mat"]
        ... )
        >>> print(f"BERTScore F1: {score.value:.4f}")
        BERTScore F1: 0.9234
    """
    
    requires_reference = True
    
    def __init__(
        self,
        model_type: str | None = None,
        lang: str | None = None,
        rescale_with_baseline: bool = True,
        device: str | None = None,
    ):
        """Initialize BERTScore metric.
        
        Args:
            model_type: Pre-trained model identifier. Popular choices:
                - "microsoft/deberta-xlarge-mnli" (recommended, large)
                - "microsoft/deberta-large-mnli" (good balance)
                - "roberta-large" (fast, good quality)
                - "bert-base-uncased" (fastest, lower quality)
            lang: Language code (e.g., "en", "zh", "fr"). If provided,
                automatically selects appropriate model.
            rescale_with_baseline: Whether to rescale scores using baseline
                (recommended for human correlation)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.name = "bertscore"
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device
        
        # Lazy import bert-score (not required for all users)
        try:
            import bert_score
            self._bert_score = bert_score
        except ImportError:
            raise ImportError(
                "bert-score is required for BERTScore metric. "
                "Install it with: pip install bert-score"
            )
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute BERTScore.
        
        Args:
            prediction: Generated text (already extracted by pipeline)
            references: List of reference texts
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with BERTScore F1 and precision/recall details
        """
        # Convert to strings
        pred_str = str(prediction)
        ref_strs = [str(ref) for ref in references]
        
        # Compute BERTScore
        # Note: bert_score.score expects lists of predictions and references
        P, R, F1 = self._bert_score.score(
            [pred_str] * len(ref_strs),  # Repeat prediction for each reference
            ref_strs,
            model_type=self.model_type,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            device=self.device,
            verbose=False,
        )
        
        # Take maximum F1 across references
        max_idx = F1.argmax().item()
        max_precision = P[max_idx].item()
        max_recall = R[max_idx].item()
        max_f1 = F1[max_idx].item()
        
        return MetricScore(
            metric_name=self.name,
            value=max_f1,  # Use F1 as primary score
            details={
                "precision": max_precision,
                "recall": max_recall,
                "f1": max_f1,
                "model_type": self.model_type or f"auto-{self.lang}",
                "num_references": len(ref_strs),
                "rescaled": self.rescale_with_baseline,
            },
            metadata=metadata or {},
        )


__all__ = ["BERTScore"]
