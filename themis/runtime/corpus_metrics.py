"""Built-in corpus-level metrics computed from persisted benchmark results."""

from __future__ import annotations

import json
from collections.abc import Sequence

from themis._optional import import_optional
from themis.records.candidate import CandidateRecord
from themis.records.trial import TrialRecord

SUPPORTED_CORPUS_METRIC_IDS: set[str] = {
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
}

_SUPPORTED_CANDIDATE_SELECTORS: set[str] = {"anchor_candidate"}
_METRIC_SPECS: dict[str, tuple[str, str | None]] = {
    "f1_micro": ("f1_score", "micro"),
    "f1_macro": ("f1_score", "macro"),
    "f1_weighted": ("f1_score", "weighted"),
    "precision_micro": ("precision_score", "micro"),
    "precision_macro": ("precision_score", "macro"),
    "precision_weighted": ("precision_score", "weighted"),
    "recall_micro": ("recall_score", "micro"),
    "recall_macro": ("recall_score", "macro"),
    "recall_weighted": ("recall_score", "weighted"),
}


def validate_candidate_selector(candidate_selector: str | None) -> str:
    """Validate the requested corpus candidate selector."""

    if candidate_selector is None:
        raise ValueError("aggregate_corpus requires a candidate_selector.")
    if candidate_selector not in _SUPPORTED_CANDIDATE_SELECTORS:
        raise ValueError(
            "Unsupported candidate_selector. This phase only supports "
            "'anchor_candidate'."
        )
    return candidate_selector


def validate_corpus_metric_id(metric_id: str) -> str:
    """Validate the requested built-in corpus metric ID."""

    if metric_id not in SUPPORTED_CORPUS_METRIC_IDS:
        raise ValueError(f"Unsupported corpus metric_id '{metric_id}'.")
    return metric_id


def select_candidate(
    trial: TrialRecord,
    *,
    candidate_selector: str,
) -> CandidateRecord | None:
    """Select the representative candidate used for one corpus metric row."""

    validate_candidate_selector(candidate_selector)
    if not trial.candidates:
        return None
    return sorted(trial.candidates, key=lambda candidate: candidate.sample_index)[0]


def prediction_label(candidate: CandidateRecord) -> str:
    """Resolve the predicted label from parsed output first, then raw text."""

    extraction = candidate.best_extraction()
    if (
        extraction is not None
        and extraction.success
        and extraction.parsed_answer is not None
    ):
        return _coerce_text(extraction.parsed_answer)
    inference = candidate.inference
    if inference is None or inference.raw_text is None:
        return ""
    return _coerce_text(inference.raw_text)


def expected_label(item_payload: object) -> str:
    """Resolve the gold label using the existing benchmark context rules."""

    return _expected_text(item_payload)


def compute_corpus_score(
    metric_id: str,
    *,
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> float:
    """Compute one built-in corpus metric from aligned gold/predicted labels."""

    validate_corpus_metric_id(metric_id)
    sklearn_metrics = import_optional("sklearn.metrics", extra="text-metrics")
    if metric_id == "cohen_kappa":
        return float(
            sklearn_metrics.cohen_kappa_score(expected_labels, predicted_labels)
        )
    scorer_name, average = _METRIC_SPECS[metric_id]
    scorer = getattr(sklearn_metrics, scorer_name)
    return float(
        scorer(
            expected_labels,
            predicted_labels,
            average=average,
            zero_division=0.0,
        )
    )


def _expected_text(context: object) -> str:
    if hasattr(context, "get"):
        for key in ("expected", "answer", "answer_letter"):
            resolved = context.get(key)  # type: ignore[attr-defined]
            if resolved is not None:
                return _coerce_text(resolved)
    return ""


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)
