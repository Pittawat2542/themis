"""CodeBLEU metric for code generation evaluation.

CodeBLEU extends BLEU with syntax awareness using abstract syntax trees (AST)
and data flow matching.

References:
    Ren et al. (2020). CodeBLEU: a Method for Automatic Evaluation of Code Synthesis.
"""

from __future__ import annotations

from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


class CodeBLEU(Metric):
    """CodeBLEU metric for code generation.
    
    CodeBLEU combines:
    - N-gram matching (like BLEU)
    - Syntax matching (AST-based)
    - Data flow matching (variable dependencies)
    
    It's more suitable for code evaluation than plain BLEU as it considers
    code structure and semantics, not just surface form.
    
    Attributes:
        name: Metric identifier ("codebleu")
        lang: Programming language ("python", "java", "javascript", etc.)
        weights: Weights for [ngram, syntax, dataflow] components
    
    Example:
        >>> from themis.evaluation.metrics.code import CodeBLEU
        >>> metric = CodeBLEU(lang="python")
        >>> score = metric.compute(
        ...     prediction="def add(a, b):\\n    return a + b",
        ...     references=["def add(x, y):\\n    return x + y"]
        ... )
        >>> print(f"CodeBLEU: {score.value:.4f}")
        CodeBLEU: 0.8234
    """
    
    requires_reference = True
    
    def __init__(
        self,
        lang: str = "python",
        weights: tuple[float, float, float] = (0.25, 0.25, 0.50),
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.50,
        theta: float = 0.0,
    ):
        """Initialize CodeBLEU metric.
        
        Args:
            lang: Programming language ("python", "java", "javascript", "go", "php", "ruby")
            weights: Weights for [ngram, weighted_ngram, syntax, dataflow].
                Default: (0.25, 0.25, 0.25, 0.25)
            alpha: Weight for n-gram matching
            beta: Weight for weighted n-gram matching
            gamma: Weight for syntax matching
            theta: Weight for data flow matching
        """
        self.name = "codebleu"
        self.lang = lang
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        
        # Lazy import codebleu (not required for all users)
        try:
            from codebleu import calc_codebleu
            self._calc_codebleu = calc_codebleu
        except ImportError:
            raise ImportError(
                "codebleu is required for CodeBLEU metric. "
                "Install it with: pip install codebleu"
            )
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute CodeBLEU score.
        
        Args:
            prediction: Generated code (already extracted by pipeline)
            references: List of reference code implementations
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with CodeBLEU value and component scores
        """
        # Convert to strings
        pred_str = str(prediction)
        ref_strs = [str(ref) for ref in references]
        
        try:
            # Compute CodeBLEU
            result = self._calc_codebleu(
                references=[ref_strs],  # List of reference lists
                predictions=[pred_str],  # List of predictions
                lang=self.lang,
                weights=(self.alpha, self.beta, self.gamma, self.theta),
            )
            
            codebleu_score = result["codebleu"]
            
            return MetricScore(
                metric_name=self.name,
                value=codebleu_score,
                details={
                    "codebleu": codebleu_score,
                    "ngram_match_score": result.get("ngram_match_score", 0.0),
                    "weighted_ngram_match_score": result.get("weighted_ngram_match_score", 0.0),
                    "syntax_match_score": result.get("syntax_match_score", 0.0),
                    "dataflow_match_score": result.get("dataflow_match_score", 0.0),
                    "lang": self.lang,
                    "num_references": len(ref_strs),
                },
                metadata=metadata or {},
            )
            
        except Exception as e:
            # Handle parsing errors (invalid code, unsupported language, etc.)
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={
                    "error": str(e),
                    "lang": self.lang,
                },
                metadata=metadata or {},
            )


__all__ = ["CodeBLEU"]
