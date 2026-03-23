"""ROUGE-based metric for RoleBench."""

from __future__ import annotations

from themis._optional import import_optional
from themis.records import MetricScore


class RoleBenchRougeMetric:
    def score(self, trial, candidate, context):
        del trial
        response_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        expected = str(context.get("expected", ""))
        if not response_text.strip() or not expected.strip():
            return MetricScore(
                metric_id="rolebench_rouge_l_f1",
                value=0.0,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            )
        try:
            rouge_scorer = import_optional(
                "rouge_score.rouge_scorer",
                extra="text-metrics",
            )
        except Exception as exc:
            return MetricScore(
                metric_id="rolebench_rouge_l_f1",
                value=0.0,
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
                error=str(exc),
            )
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(expected, str(response_text))["rougeL"]
        return MetricScore(
            metric_id="rolebench_rouge_l_f1",
            value=float(result.fmeasure),
            details={
                "precision": float(result.precision),
                "recall": float(result.recall),
                "f1": float(result.fmeasure),
            },
        )
