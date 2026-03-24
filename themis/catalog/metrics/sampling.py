"""Sampling-aware metric helpers and trial-level metric implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import statistics
from typing import Protocol, cast

from themis.contracts.protocols import Metric
from themis.records import CandidateRecord, MetricScore
from themis.specs.foundational import MetricRefSpec
from themis.types.json_validation import validate_json_dict

_CONFIDENCE_PATTERN = re.compile(
    r"confidence\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<pct>%?)",
    re.IGNORECASE,
)


class _TextMetric(Protocol):
    def __call__(self, candidate: str, reference: str | None) -> float: ...


@dataclass(frozen=True, slots=True)
class SelfConsistencyResult:
    """Selected answer plus agreement statistics for one candidate set."""

    value: float
    selected_answer: str
    agreement_ratio: float
    candidate_scores: list[float]
    bucket_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class BestOfNResult:
    """Best candidate selection result for one candidate set."""

    value: float
    best_candidate: str
    best_score: float
    rankings: list[tuple[str, float]]


def self_consistency(
    *,
    candidates: Sequence[str],
    reference: str | None,
    metric: _TextMetric,
    strategy: str = "majority_vote",
) -> SelfConsistencyResult:
    """Run self-consistency selection over raw string candidates."""

    _require_candidates(candidates)
    candidate_scores = [float(metric(candidate, reference)) for candidate in candidates]
    winning = _select_text_bucket(candidates, strategy=strategy)
    agreement_ratio = winning.weight / len(candidates)
    return SelfConsistencyResult(
        value=float(metric(winning.answer, reference)),
        selected_answer=winning.answer,
        agreement_ratio=agreement_ratio,
        candidate_scores=candidate_scores,
        bucket_weights=winning.bucket_weights,
    )


def best_of_n(
    *,
    candidates: Sequence[str],
    reference: str | None,
    metric: _TextMetric,
    return_ranking: bool = False,
) -> BestOfNResult:
    """Select the best candidate under a scoring callable."""

    _require_candidates(candidates)
    rankings = sorted(
        ((candidate, float(metric(candidate, reference))) for candidate in candidates),
        key=lambda item: (-item[1], candidates.index(item[0]), item[0]),
    )
    best_candidate, best_score = rankings[0]
    return BestOfNResult(
        value=best_score,
        best_candidate=best_candidate,
        best_score=best_score,
        rankings=rankings if return_ranking else [],
    )


def pass_at_k(*, total_samples: int, correct_samples: int, k: int) -> float:
    """Compute the unbiased pass@k estimator from Chen et al. (2021)."""

    _require_k(k=k, total=total_samples)
    if correct_samples < 0 or correct_samples > total_samples:
        raise ValueError("correct_samples must be between 0 and total_samples.")
    if total_samples - correct_samples < k:
        return 1.0
    return 1.0 - (
        math.comb(total_samples - correct_samples, k) / math.comb(total_samples, k)
    )


def avg_at_k(
    scores: Sequence[float],
    *,
    k: int,
    mode: str = "top_k",
) -> float:
    """Average either the top-k or first-k scores."""

    _require_k(k=k, total=len(scores))
    selected = (
        list(scores[:k]) if mode == "first_k" else sorted(scores, reverse=True)[:k]
    )
    if mode not in {"top_k", "first_k"}:
        raise ValueError("mode must be either 'top_k' or 'first_k'.")
    return sum(selected) / len(selected)


def majority_at_k(
    *,
    candidates: Sequence[str],
    reference: str | None,
    metric: _TextMetric,
    k: int,
) -> float:
    """Score the majority-voted answer among the first k samples."""

    _require_k(k=k, total=len(candidates))
    winning = _select_text_bucket(candidates[:k], strategy="majority_vote")
    return float(metric(winning.answer, reference))


def variance_at_k(scores: Sequence[float], *, k: int) -> float:
    """Return the population variance across the first k scores."""

    _require_k(k=k, total=len(scores))
    if k == 1:
        return 0.0
    return float(statistics.pvariance(scores[:k]))


def acc_consistency(
    *,
    candidates: Sequence[str],
    reference: str | None,
    metric: _TextMetric,
) -> float:
    """Return 1 when all candidates agree and that answer is correct."""

    _require_candidates(candidates)
    winning = _select_text_bucket(candidates, strategy="majority_vote")
    if winning.count != len(candidates):
        return 0.0
    return float(metric(winning.answer, reference))


class SelfConsistencyMetric:
    """Trial-level self-consistency metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        strategy = _metric_config_str(
            context,
            key="strategy",
            default="majority_vote",
        )
        candidate_scores = _base_metric_scores(trial, candidates, context)
        winning = _select_candidate_bucket(candidates, strategy=strategy)
        representative = winning.representative
        value = _score_single_candidate(
            trial,
            representative,
            context,
        )
        return MetricScore(
            metric_id="self_consistency",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "strategy": strategy,
                    "selected_answer": winning.answer,
                    "selected_candidate_id": (
                        representative.candidate_id or representative.spec_hash
                    ),
                    "agreement_ratio": winning.weight / len(candidates),
                    "candidate_scores": candidate_scores,
                    "bucket_weights": winning.bucket_weights,
                    "candidate_ids": [
                        candidate.candidate_id or candidate.spec_hash
                        for candidate in candidates
                    ],
                },
                label="self_consistency details",
            ),
        )


class BestOfNMetric:
    """Trial-level best-of-n metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        scores = [
            _RankedCandidate(
                candidate=candidate,
                candidate_id=candidate.candidate_id or candidate.spec_hash,
                score=_score_single_candidate(trial, candidate, context),
                answer=_candidate_answer(candidate),
                sample_index=candidate.sample_index,
            )
            for candidate in candidates
        ]
        ranked = sorted(
            scores,
            key=lambda item: (
                -item.score,
                item.sample_index,
                item.candidate_id,
            ),
        )
        details: dict[str, object] = {
            "metric_level": "trial",
            "best_candidate_id": ranked[0].candidate_id,
            "best_answer": ranked[0].answer,
        }
        if _metric_config_bool(context, key="return_ranking", default=False):
            details["rankings"] = [
                {
                    "candidate_id": item.candidate_id,
                    "score": item.score,
                    "answer": item.answer,
                }
                for item in ranked
            ]
        return MetricScore(
            metric_id="best_of_n",
            value=ranked[0].score,
            details=validate_json_dict(details, label="best_of_n details"),
        )


class PassAtKMetric:
    """Trial-level pass@k metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        k = _metric_config_int(context, key="k")
        threshold = _metric_config_float(context, key="success_threshold", default=1.0)
        scores = _base_metric_scores(trial, candidates, context)
        correct = sum(score >= threshold for score in scores)
        value = pass_at_k(total_samples=len(candidates), correct_samples=correct, k=k)
        return MetricScore(
            metric_id="pass_at_k",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "k": k,
                    "correct_samples": correct,
                    "total_samples": len(candidates),
                    "success_threshold": threshold,
                    "candidate_scores": scores,
                },
                label="pass_at_k details",
            ),
        )


class AvgAtKMetric:
    """Trial-level avg@k metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        k = _metric_config_int(context, key="k")
        mode = _metric_config_str(context, key="mode", default="top_k")
        scores = _base_metric_scores(trial, candidates, context)
        value = avg_at_k(scores, k=k, mode=mode)
        return MetricScore(
            metric_id="avg_at_k",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "k": k,
                    "mode": mode,
                    "candidate_scores": scores,
                },
                label="avg_at_k details",
            ),
        )


class AccConsistencyMetric:
    """Trial-level joint accuracy and consistency metric."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        winning = _select_candidate_bucket(candidates, strategy="majority_vote")
        if winning.count != len(candidates):
            return MetricScore(
                metric_id="acc_consistency",
                value=0.0,
                details=validate_json_dict(
                    {
                        "metric_level": "trial",
                        "consistent": False,
                        "selected_answer": winning.answer,
                    },
                    label="acc_consistency details",
                ),
            )
        value = _score_single_candidate(trial, winning.representative, context)
        return MetricScore(
            metric_id="acc_consistency",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "consistent": True,
                    "selected_answer": winning.answer,
                },
                label="acc_consistency details",
            ),
        )


class MajorityAtKMetric:
    """Trial-level majority@k metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        k = _metric_config_int(context, key="k")
        _require_k(k=k, total=len(candidates))
        winning = _select_candidate_bucket(
            list(candidates[:k]),
            strategy="majority_vote",
        )
        value = _score_single_candidate(trial, winning.representative, context)
        return MetricScore(
            metric_id="majority_at_k",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "k": k,
                    "majority_answer": winning.answer,
                    "selected_candidate_id": (
                        winning.representative.candidate_id
                        or winning.representative.spec_hash
                    ),
                },
                label="majority_at_k details",
            ),
        )


class VarianceAtKMetric:
    """Trial-level score variance metric over all candidates."""

    def score_trial(
        self,
        trial,
        candidates: Sequence[CandidateRecord],
        context: Mapping[str, object],
    ) -> MetricScore:
        _require_candidate_records(candidates)
        k = _metric_config_int(context, key="k")
        scores = _base_metric_scores(trial, candidates, context)
        value = variance_at_k(scores, k=k)
        return MetricScore(
            metric_id="variance_at_k",
            value=value,
            details=validate_json_dict(
                {
                    "metric_level": "trial",
                    "k": k,
                    "candidate_scores": scores,
                },
                label="variance_at_k details",
            ),
        )


@dataclass(frozen=True, slots=True)
class _BucketSelection:
    answer: str
    count: int
    weight: float
    representative: CandidateRecord
    bucket_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class _TextBucketSelection:
    answer: str
    count: int
    weight: float
    bucket_weights: dict[str, float]


@dataclass(slots=True)
class _CandidateVoteBucket:
    answer: str
    count: int
    weight: float
    representative: CandidateRecord


@dataclass(slots=True)
class _TextVoteBucket:
    answer: str
    count: int
    weight: float
    index: int


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    candidate: CandidateRecord
    candidate_id: str
    score: float
    answer: str
    sample_index: int


def _select_candidate_bucket(
    candidates: Sequence[CandidateRecord],
    *,
    strategy: str,
) -> _BucketSelection:
    if strategy not in {"majority_vote", "weighted_vote", "modal_answer"}:
        raise ValueError(
            "strategy must be one of: majority_vote, weighted_vote, modal_answer."
        )
    buckets: dict[str, _CandidateVoteBucket] = {}
    for candidate in candidates:
        answer = _candidate_answer(candidate)
        normalized = _normalize_answer(answer)
        bucket = buckets.setdefault(
            normalized,
            _CandidateVoteBucket(
                answer=answer,
                count=0,
                weight=0.0,
                representative=candidate,
            ),
        )
        bucket.count += 1
        bucket.weight += (
            _candidate_confidence(candidate) if strategy == "weighted_vote" else 1.0
        )
        if candidate.sample_index < bucket.representative.sample_index:
            bucket.representative = candidate
            bucket.answer = answer
    winning = sorted(
        buckets.values(),
        key=lambda item: (
            -item.weight,
            -item.count,
            item.representative.sample_index,
            item.answer,
        ),
    )[0]
    return _BucketSelection(
        answer=winning.answer,
        count=winning.count,
        weight=winning.weight,
        representative=winning.representative,
        bucket_weights={item.answer: item.weight for item in buckets.values()},
    )


def _select_text_bucket(
    candidates: Sequence[str],
    *,
    strategy: str,
) -> _TextBucketSelection:
    if strategy not in {"majority_vote", "weighted_vote", "modal_answer"}:
        raise ValueError(
            "strategy must be one of: majority_vote, weighted_vote, modal_answer."
        )
    buckets: dict[str, _TextVoteBucket] = {}
    for index, answer in enumerate(candidates):
        normalized = _normalize_answer(answer)
        bucket = buckets.setdefault(
            normalized,
            _TextVoteBucket(answer=answer, count=0, weight=0.0, index=index),
        )
        bucket.count += 1
        bucket.weight += 1.0
        if index < bucket.index:
            bucket.answer = answer
            bucket.index = index
    winning = sorted(
        buckets.values(),
        key=lambda item: (
            -item.weight,
            -item.count,
            item.index,
            item.answer,
        ),
    )[0]
    return _TextBucketSelection(
        answer=winning.answer,
        count=winning.count,
        weight=winning.weight,
        bucket_weights={item.answer: item.weight for item in buckets.values()},
    )


def _candidate_answer(candidate: CandidateRecord) -> str:
    extraction = candidate.best_extraction()
    if (
        extraction is not None
        and extraction.success
        and extraction.parsed_answer is not None
    ):
        return str(extraction.parsed_answer)
    inference = getattr(candidate, "inference", None)
    raw_text = getattr(inference, "raw_text", "") if inference is not None else ""
    return str(raw_text)


def _candidate_confidence(candidate: CandidateRecord) -> float:
    extraction = candidate.best_extraction()
    parsed_answer = (
        extraction.parsed_answer
        if extraction is not None and extraction.success
        else None
    )
    if isinstance(parsed_answer, Mapping):
        confidence = parsed_answer.get("confidence")
        normalized = _normalize_confidence(confidence)
        if normalized is not None:
            return normalized
    inference = getattr(candidate, "inference", None)
    raw_text = str(getattr(inference, "raw_text", "") or "")
    match = _CONFIDENCE_PATTERN.search(raw_text)
    if match is None:
        return 1.0
    value = float(match.group("value"))
    if match.group("pct"):
        return max(0.0, min(1.0, value / 100.0))
    return max(0.0, min(1.0, value))


def _normalize_confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    normalized = float(value)
    if normalized > 1.0:
        normalized /= 100.0
    return max(0.0, min(1.0, normalized))


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def _base_metric_scores(
    trial,
    candidates: Sequence[CandidateRecord],
    context: Mapping[str, object],
) -> list[float]:
    return [
        _score_single_candidate(trial, candidate, context) for candidate in candidates
    ]


def _score_single_candidate(
    trial,
    candidate: CandidateRecord,
    context: Mapping[str, object],
) -> float:
    metric_ref = _base_metric_ref(context)
    registry = context.get("metric_registry")
    if registry is None or not hasattr(registry, "get_metric"):
        raise ValueError("Trial metric requires 'metric_registry' in context.")
    metric = registry.get_metric(metric_ref.id)
    if not isinstance(metric, Metric):
        raise ValueError("base_metric must resolve to a candidate-level metric.")
    nested_context = dict(context)
    nested_context["metric_config"] = dict(metric_ref.config)
    score_record = cast(Metric, metric).score(trial, candidate, nested_context)
    return float(score_record.value)


def _base_metric_ref(context: Mapping[str, object]) -> MetricRefSpec:
    config = context.get("metric_config")
    if not isinstance(config, Mapping):
        raise ValueError("Trial metric requires mapping metric_config.")
    base_metric = config.get("base_metric")
    if isinstance(base_metric, MetricRefSpec):
        return base_metric
    if isinstance(base_metric, str):
        return MetricRefSpec(id=base_metric)
    if isinstance(base_metric, Mapping):
        return MetricRefSpec.model_validate(dict(base_metric))
    raise ValueError("Trial metric requires a 'base_metric' metric ref.")


def _metric_config_str(
    context: Mapping[str, object],
    *,
    key: str,
    default: str,
) -> str:
    config = context.get("metric_config")
    if not isinstance(config, Mapping):
        return default
    value = config.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metric_config['{key}'] must be a non-empty string.")
    return value


def _metric_config_int(context: Mapping[str, object], *, key: str) -> int:
    config = context.get("metric_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Trial metric requires integer metric_config['{key}'].")
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metric_config['{key}'] must be an integer.")
    return value


def _metric_config_float(
    context: Mapping[str, object],
    *,
    key: str,
    default: float,
) -> float:
    config = context.get("metric_config")
    if not isinstance(config, Mapping):
        return default
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric_config['{key}'] must be numeric.")
    return float(value)


def _metric_config_bool(
    context: Mapping[str, object],
    *,
    key: str,
    default: bool,
) -> bool:
    config = context.get("metric_config")
    if not isinstance(config, Mapping):
        return default
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"metric_config['{key}'] must be a boolean.")
    return value


def _require_candidates(candidates: Sequence[str]) -> None:
    if not candidates:
        raise ValueError("candidates must contain at least one value.")


def _require_candidate_records(candidates: Sequence[CandidateRecord]) -> None:
    if not candidates:
        raise ValueError("candidates must contain at least one candidate record.")


def _require_k(*, k: int, total: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1.")
    if total < 1:
        raise ValueError("total must be >= 1.")
    if k > total:
        raise ValueError("k must be <= the number of available samples.")
