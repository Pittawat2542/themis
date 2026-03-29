from __future__ import annotations

import pytest

from themis.core.models import GenerationResult
from themis.core.subjects import (
    CandidateSetSubject,
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
