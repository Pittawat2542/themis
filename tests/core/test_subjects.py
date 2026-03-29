from __future__ import annotations

import pytest

from themis.core.models import GenerationResult
from themis.core import subjects as subjects_module
from themis.core.subjects import (
    CandidateSetSubject,
    candidate_set_subject_for_llm_metric,
    candidate_set_subject_for_selection_metric,
    validate_candidate_set_for_llm_metric,
    validate_candidate_set_for_selection_metric,
)


def test_candidate_subject_reports_size() -> None:
    subject = CandidateSetSubject(
        candidates=[
            GenerationResult(candidate_id="candidate-1", final_output="4"),
            GenerationResult(candidate_id="candidate-2", final_output="4"),
        ]
    )

    assert subject.size == 2


def test_llm_metric_subject_validation_requires_exactly_one_candidate() -> None:
    single = CandidateSetSubject(
        candidates=[GenerationResult(candidate_id="candidate-1", final_output="4")]
    )
    pair = CandidateSetSubject(
        candidates=[
            GenerationResult(candidate_id="candidate-1", final_output="4"),
            GenerationResult(candidate_id="candidate-2", final_output="4"),
        ]
    )

    validate_candidate_set_for_llm_metric(single)

    with pytest.raises(ValueError, match="exactly one"):
        validate_candidate_set_for_llm_metric(pair)


def test_selection_metric_subject_validation_requires_multiple_candidates() -> None:
    single = CandidateSetSubject(
        candidates=[GenerationResult(candidate_id="candidate-1", final_output="4")]
    )
    pair = CandidateSetSubject(
        candidates=[
            GenerationResult(candidate_id="candidate-1", final_output="4"),
            GenerationResult(candidate_id="candidate-2", final_output="4"),
        ]
    )

    validate_candidate_set_for_selection_metric(pair)

    with pytest.raises(ValueError, match="at least two"):
        validate_candidate_set_for_selection_metric(single)


def test_llm_metric_subject_factory_calls_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_validate(subject: CandidateSetSubject) -> None:
        nonlocal called
        called = True
        assert subject.size == 1

    monkeypatch.setattr(subjects_module, "validate_candidate_set_for_llm_metric", fake_validate)

    subject = candidate_set_subject_for_llm_metric(
        [GenerationResult(candidate_id="candidate-1", final_output="4")]
    )

    assert called is True
    assert subject.size == 1


def test_selection_metric_subject_factory_calls_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_validate(subject: CandidateSetSubject) -> None:
        nonlocal called
        called = True
        assert subject.size == 2

    monkeypatch.setattr(
        subjects_module,
        "validate_candidate_set_for_selection_metric",
        fake_validate,
    )

    subject = candidate_set_subject_for_selection_metric(
        [
            GenerationResult(candidate_id="candidate-1", final_output="4"),
            GenerationResult(candidate_id="candidate-2", final_output="5"),
        ]
    )

    assert called is True
    assert subject.size == 2
