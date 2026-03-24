"""Candidate-level NLP metric wrappers backed by optional libraries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Literal

from themis._optional import import_optional
from themis.errors import MetricError
from themis.records import MetricScore
from themis.types.enums import ErrorCode
from themis.types.json_validation import validate_json_dict


class BleuMetric:
    """Sentence-level BLEU wrapper backed by NLTK."""

    def score(self, trial, candidate, context):
        del trial
        nltk_bleu = _import_metric_module(
            "nltk.translate.bleu_score",
            metric_id="bleu",
        )
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("bleu")
        value = nltk_bleu.sentence_bleu(
            [reference.split() for reference in references],
            candidate_text.split(),
        )
        return MetricScore(metric_id="bleu", value=float(value))


class RougeMetric:
    """ROUGE wrapper for one configured variant."""

    def __init__(self, variant: Literal["rouge1", "rouge2", "rougeL"]) -> None:
        self._variant = variant
        self._metric_id = variant.replace("rouge", "rouge_").lower()

    def score(self, trial, candidate, context):
        del trial
        rouge_scorer = _import_metric_module(
            "rouge_score.rouge_scorer",
            metric_id=self._metric_id,
        )
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score(self._metric_id)
        scorer = rouge_scorer.RougeScorer([self._variant], use_stemmer=True)
        scored = [
            scorer.score(reference, candidate_text)[self._variant]
            for reference in references
        ]
        best = max(scored, key=lambda item: item.fmeasure)
        return MetricScore(
            metric_id=self._metric_id,
            value=float(best.fmeasure),
            details=validate_json_dict(
                {
                    "precision": float(best.precision),
                    "recall": float(best.recall),
                    "f1": float(best.fmeasure),
                },
                label=f"{self._metric_id} details",
            ),
        )


class MeteorMetric:
    """METEOR wrapper backed by NLTK."""

    def score(self, trial, candidate, context):
        del trial
        meteor = _import_metric_module(
            "nltk.translate.meteor_score",
            metric_id="meteor",
        )
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("meteor")
        value = meteor.meteor_score(
            [reference.split() for reference in references],
            candidate_text.split(),
        )
        return MetricScore(metric_id="meteor", value=float(value))


class BERTScoreMetric:
    """BERTScore wrapper backed by ``bert-score``."""

    def score(self, trial, candidate, context):
        del trial
        bert_score = _bert_score_score()
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("bertscore")
        precision, recall, f1 = bert_score(
            [candidate_text],
            [references[0]],
            lang="en",
            verbose=False,
        )
        return MetricScore(
            metric_id="bertscore",
            value=float(f1[0].item()),
            details=validate_json_dict(
                {
                    "precision": float(precision[0].item()),
                    "recall": float(recall[0].item()),
                    "f1": float(f1[0].item()),
                },
                label="bertscore details",
            ),
        )


class SacreBleuMetric:
    """SacreBLEU wrapper backed by ``sacrebleu``."""

    def score(self, trial, candidate, context):
        del trial
        sacrebleu = _import_metric_module("sacrebleu", metric_id="sacrebleu")
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("sacrebleu")
        value = sacrebleu.sentence_bleu(candidate_text, references).score / 100.0
        return MetricScore(metric_id="sacrebleu", value=float(value))


class ChrFMetric:
    """chrF wrapper backed by ``sacrebleu``."""

    def score(self, trial, candidate, context):
        del trial
        sacrebleu = _import_metric_module("sacrebleu", metric_id="chrf")
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("chrf")
        scorer = sacrebleu.metrics.CHRF()
        value = scorer.sentence_score(candidate_text, references).score / 100.0
        return MetricScore(metric_id="chrf", value=float(value))


class TERMetric:
    """TER wrapper backed by ``sacrebleu``."""

    def score(self, trial, candidate, context):
        del trial
        sacrebleu = _import_metric_module("sacrebleu", metric_id="ter")
        references = _reference_texts(context)
        candidate_text = _candidate_text(candidate)
        if not candidate_text.strip() or not any(text.strip() for text in references):
            return _zero_score("ter")
        scorer = sacrebleu.metrics.TER()
        value = scorer.sentence_score(candidate_text, references).score / 100.0
        return MetricScore(metric_id="ter", value=float(value))


class EditDistanceMetric:
    """Levenshtein edit-distance wrapper backed by ``rapidfuzz``."""

    def score(self, trial, candidate, context):
        del trial
        rapidfuzz_distance = _import_metric_module(
            "rapidfuzz.distance.Levenshtein",
            metric_id="edit_distance",
        )
        expected = str(context.get("expected", context.get("reference", "")))
        actual = _candidate_text(candidate)
        return MetricScore(
            metric_id="edit_distance",
            value=float(rapidfuzz_distance.distance(actual, expected)),
        )


def _candidate_text(candidate: object) -> str:
    inference = getattr(candidate, "inference", None)
    return str(getattr(inference, "raw_text", "") or "")


def _reference_texts(context: Mapping[str, object]) -> list[str]:
    references = context.get("references")
    if isinstance(references, Sequence) and not isinstance(references, str):
        return [str(reference) for reference in references]
    return [str(context.get("expected", context.get("reference", "")))]


def _zero_score(metric_id: str) -> MetricScore:
    return MetricScore(metric_id=metric_id, value=0.0)


def _import_metric_module(module_name: str, *, metric_id: str):
    try:
        return import_optional(module_name, extra="text-metrics")
    except Exception as exc:
        raise MetricError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=str(exc),
            details={"metric_id": metric_id},
        ) from exc


@lru_cache(maxsize=1)
def _bert_score_score():
    module = _import_metric_module("bert_score", metric_id="bertscore")
    return module.score
