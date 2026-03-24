from __future__ import annotations

import pytest

from themis.catalog.metrics import (
    AvgAtKMetric,
    BestOfNMetric,
    PassAtKMetric,
    SelfConsistencyMetric,
    avg_at_k,
    best_of_n,
    pass_at_k,
    self_consistency,
)
from themis.catalog.runtime.registry import register_catalog_metrics
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord
from themis.registry.plugin_registry import PluginRegistry


def _candidate(text: str, *, sample_index: int) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"candidate-{sample_index}",
        spec_hash=f"candidate-{sample_index}",
        sample_index=sample_index,
        inference=InferenceRecord(
            spec_hash=f"inference-{sample_index}",
            raw_text=text,
        ),
    )


class _NormalizedExactMatch:
    def score(self, trial, candidate, context):
        del trial
        actual = str(candidate.inference.raw_text).strip().lower()
        expected = str(context.get("expected", "")).strip().lower()
        return MetricScore(
            metric_id="normalized_exact_match", value=float(actual == expected)
        )


def test_self_consistency_helper_returns_majority_vote_details() -> None:
    result = self_consistency(
        candidates=["Paris", "paris", "London"],
        reference="Paris",
        metric=lambda candidate, reference: float(
            candidate.strip().lower() == reference.strip().lower()
        ),
        strategy="majority_vote",
    )

    assert result.value == 1.0
    assert result.selected_answer.lower() == "paris"
    assert result.agreement_ratio == pytest.approx(2 / 3)
    assert result.candidate_scores == [1.0, 1.0, 0.0]


def test_best_of_n_helper_returns_ranking() -> None:
    result = best_of_n(
        candidates=["a", "bbb", "cc"],
        reference="",
        metric=lambda candidate, reference: float(len(candidate)),
        return_ranking=True,
    )

    assert result.value == 3.0
    assert result.best_candidate == "bbb"
    assert result.rankings == [("bbb", 3.0), ("cc", 2.0), ("a", 1.0)]


def test_pass_at_k_helper_uses_unbiased_estimator() -> None:
    assert pass_at_k(total_samples=5, correct_samples=2, k=1) == pytest.approx(0.4)


def test_avg_at_k_helper_supports_top_k_and_first_k_modes() -> None:
    assert avg_at_k([0.1, 0.9, 0.4], k=2, mode="top_k") == pytest.approx(0.65)
    assert avg_at_k([0.1, 0.9, 0.4], k=2, mode="first_k") == pytest.approx(0.5)


def test_trial_metrics_score_against_metric_refs() -> None:
    registry = PluginRegistry()
    register_catalog_metrics(registry)
    registry.register_metric("normalized_exact_match", _NormalizedExactMatch())
    candidates = [
        _candidate("Paris", sample_index=0),
        _candidate("paris", sample_index=1),
        _candidate("London", sample_index=2),
    ]
    context = {
        "expected": "Paris",
        "metric_registry": registry,
        "metric_config": {"base_metric": {"id": "normalized_exact_match"}},
        "anchor_candidate": candidates[0],
    }

    self_consistency_score = SelfConsistencyMetric().score_trial(
        None, candidates, context
    )
    best_of_n_score = BestOfNMetric().score_trial(
        None,
        candidates,
        context
        | {
            "metric_config": {
                "base_metric": {"id": "normalized_exact_match"},
                "return_ranking": True,
            }
        },
    )
    avg_score = AvgAtKMetric().score_trial(
        None,
        candidates,
        context
        | {"metric_config": {"base_metric": {"id": "normalized_exact_match"}, "k": 2}},
    )
    pass_score = PassAtKMetric().score_trial(
        None,
        candidates,
        context
        | {"metric_config": {"base_metric": {"id": "normalized_exact_match"}, "k": 2}},
    )

    assert self_consistency_score.value == 1.0
    assert self_consistency_score.details["selected_answer"] == "Paris"
    assert best_of_n_score.value == 1.0
    assert best_of_n_score.details["best_candidate_id"] == "candidate-0"
    assert avg_score.value == 1.0
    assert pass_score.value == 1.0


def test_register_catalog_metrics_registers_sampling_metrics() -> None:
    registry = PluginRegistry()

    register_catalog_metrics(registry)

    for metric_id in {
        "self_consistency",
        "best_of_n",
        "pass_at_k",
        "avg_at_k",
        "acc_consistency",
        "majority_at_k",
        "variance_at_k",
        "bleu",
        "rouge_l",
        "meteor",
        "bertscore",
        "sacrebleu",
        "chrf",
        "ter",
        "edit_distance",
    }:
        assert registry.has_metric(metric_id), metric_id


def test_register_catalog_metrics_excludes_corpus_classification_metrics() -> None:
    registry = PluginRegistry()

    register_catalog_metrics(registry)

    for metric_id in {
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "cohen_kappa",
    }:
        assert not registry.has_metric(metric_id), metric_id
