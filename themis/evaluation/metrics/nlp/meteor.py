"""METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.

METEOR is an MT evaluation metric that addresses some weaknesses of BLEU by
incorporating stemming, synonymy, and explicit word ordering.

References:
    Banerjee & Lavie (2005). METEOR: An Automatic Metric for MT Evaluation
    with Improved Correlation with Human Judgments.
"""

from __future__ import annotations

from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


class METEOR(Metric):
    """METEOR metric using nltk library.
    
    METEOR compares generated text to references using:
    - Exact word matching
    - Stemming (using Porter stemmer)
    - Synonymy (using WordNet)
    - Word order (using chunk matching)
    
    It computes a weighted F-score with emphasis on recall and applies a penalty
    for word order differences.
    
    Attributes:
        name: Metric identifier ("meteor")
        alpha: Weight for precision vs recall (default: 0.9, favors recall)
        beta: Weight for fragmentation penalty (default: 3.0)
        gamma: Fragmentation penalty coefficient (default: 0.5)
    
    Example:
        >>> from themis.evaluation.metrics.nlp import METEOR
        >>> metric = METEOR()
        >>> score = metric.compute(
        ...     prediction="The cat sat on the mat",
        ...     references=["The cat is on the mat", "A cat sits on a mat"]
        ... )
        >>> print(f"METEOR: {score.value:.4f}")
        METEOR: 0.8234
    """
    
    requires_reference = True
    
    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
    ):
        """Initialize METEOR metric.
        
        Args:
            alpha: Weight for precision vs recall (0-1). Higher values favor recall.
                Default 0.9 emphasizes recall like original METEOR.
            beta: Weight for fragmentation penalty (typically 3.0)
            gamma: Fragmentation penalty coefficient (typically 0.5)
        """
        self.name = "meteor"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Lazy import nltk (not required for all users)
        try:
            from nltk.translate import meteor_score as meteor
            self._meteor = meteor
            
            # Download required NLTK data if not present
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading WordNet data for METEOR...")
                nltk.download('wordnet', quiet=True)
            
            try:
                nltk.data.find('omw-1.4')
            except LookupError:
                print("Downloading OMW data for METEOR...")
                nltk.download('omw-1.4', quiet=True)
                
        except ImportError:
            raise ImportError(
                "nltk is required for METEOR metric. "
                "Install it with: pip install nltk"
            )
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute METEOR score.
        
        Args:
            prediction: Generated text (already extracted by pipeline)
            references: List of reference texts
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with METEOR value (0.0-1.0)
        """
        # Convert to strings and tokenize
        pred_str = str(prediction)
        ref_strs = [str(ref) for ref in references]
        
        # Tokenize (simple whitespace tokenization)
        pred_tokens = pred_str.split()
        ref_tokens_list = [ref.split() for ref in ref_strs]
        
        # Compute METEOR score
        # Note: nltk's meteor_score takes one reference at a time
        # We compute for each reference and take the maximum
        max_score = 0.0
        
        for ref_tokens in ref_tokens_list:
            try:
                score = self._meteor.meteor_score(
                    [ref_tokens],  # References should be list of tokenized references
                    pred_tokens,   # Hypothesis is tokenized prediction
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                max_score = max(max_score, score)
            except Exception as e:
                # Handle edge cases (empty strings, etc.)
                print(f"Warning: METEOR computation failed: {e}")
                continue
        
        return MetricScore(
            metric_name=self.name,
            value=max_score,
            details={
                "meteor_score": max_score,
                "num_references": len(ref_strs),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            metadata=metadata or {},
        )


__all__ = ["METEOR"]
