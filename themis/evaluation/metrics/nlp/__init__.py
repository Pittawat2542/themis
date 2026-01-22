"""NLP evaluation metrics.

This module provides standard NLP metrics for text generation evaluation:
- BLEU: Bilingual Evaluation Understudy for translation quality
- ROUGE: Recall-Oriented Understudy for Gisting Evaluation for summarization
- BERTScore: Contextual embeddings-based evaluation
- METEOR: Metric for Evaluation of Translation with Explicit ORdering
"""

from themis.evaluation.metrics.nlp.bleu import BLEU
from themis.evaluation.metrics.nlp.rouge import ROUGE, ROUGEVariant
from themis.evaluation.metrics.nlp.bertscore import BERTScore
from themis.evaluation.metrics.nlp.meteor import METEOR

__all__ = [
    "BLEU",
    "ROUGE",
    "ROUGEVariant",
    "BERTScore",
    "METEOR",
]
