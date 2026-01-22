"""BLEU (Bilingual Evaluation Understudy) metric implementation.

BLEU measures the similarity between generated text and reference translations
using n-gram precision with brevity penalty.

References:
    Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
"""

from __future__ import annotations

from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


class BLEU(Metric):
    """BLEU metric using sacrebleu library.
    
    BLEU is a precision-based metric that computes n-gram overlap between
    generated text and reference translations. It includes a brevity penalty
    to penalize short translations.
    
    Attributes:
        name: Metric identifier ("bleu")
        lowercase: Whether to lowercase text before scoring
        tokenize: Tokenization method ("13a", "intl", "zh", "ja-mecab", etc.)
        max_ngram_order: Maximum n-gram order (default: 4)
    
    Example:
        >>> from themis.evaluation.metrics.nlp import BLEU
        >>> metric = BLEU()
        >>> score = metric.compute(
        ...     prediction="The cat sat on the mat",
        ...     references=["The cat is on the mat", "A cat is sitting on a mat"]
        ... )
        >>> print(f"BLEU: {score.value:.4f}")
        BLEU: 0.4523
    """
    
    requires_reference = True
    
    def __init__(
        self,
        lowercase: bool = False,
        tokenize: str = "13a",
        max_ngram_order: int = 4,
    ):
        """Initialize BLEU metric.
        
        Args:
            lowercase: Convert text to lowercase before scoring
            tokenize: Tokenization method:
                - "13a": Default Moses tokenizer (punctuation split)
                - "intl": International tokenizer
                - "zh": Chinese tokenizer
                - "ja-mecab": Japanese MeCab tokenizer
                - "none": No tokenization
            max_ngram_order: Maximum n-gram order (typically 4)
        """
        self.name = "bleu"
        self.lowercase = lowercase
        self.tokenize = tokenize
        self.max_ngram_order = max_ngram_order
        
        # Lazy import sacrebleu (not required for all users)
        try:
            from sacrebleu import BLEU as SacreBLEU
            self._scorer = SacreBLEU(
                lowercase=lowercase,
                tokenize=tokenize,
                max_ngram_order=max_ngram_order,
            )
        except ImportError:
            raise ImportError(
                "sacrebleu is required for BLEU metric. "
                "Install it with: pip install sacrebleu"
            )
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute BLEU score.
        
        Args:
            prediction: Generated text (already extracted by pipeline)
            references: List of reference translations
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with BLEU value (0.0-1.0) and detailed scores
        """
        # Convert to strings
        pred_str = str(prediction)
        ref_strs = [str(ref) for ref in references]
        
        # Compute BLEU score
        score_obj = self._scorer.sentence_score(pred_str, ref_strs)
        
        # Extract scores (sacrebleu returns 0-100, we normalize to 0-1)
        bleu_score = score_obj.score / 100.0
        
        # Extract precision scores for each n-gram
        precisions = [p / 100.0 for p in score_obj.precisions]
        
        return MetricScore(
            metric_name=self.name,
            value=bleu_score,
            details={
                "bleu_score": bleu_score,
                "precision_1": precisions[0] if len(precisions) > 0 else 0.0,
                "precision_2": precisions[1] if len(precisions) > 1 else 0.0,
                "precision_3": precisions[2] if len(precisions) > 2 else 0.0,
                "precision_4": precisions[3] if len(precisions) > 3 else 0.0,
                "brevity_penalty": score_obj.bp,
                "length_ratio": score_obj.sys_len / score_obj.ref_len if score_obj.ref_len > 0 else 0.0,
                "sys_len": score_obj.sys_len,
                "ref_len": score_obj.ref_len,
            },
            metadata=metadata or {},
        )


__all__ = ["BLEU"]
