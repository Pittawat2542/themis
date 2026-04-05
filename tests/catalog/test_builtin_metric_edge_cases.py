from __future__ import annotations

from typing import cast

from themis.catalog import load
from themis.core.contexts import ScoreContext
from themis.core.models import Case, ParsedOutput, Score
from themis.core.protocols import PureMetric


def test_exact_match_metric_treats_type_mismatch_as_failure() -> None:
    metric = cast(PureMetric, load("builtin/exact_match"))
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output={"answer": "4"},
    )
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_output=ParsedOutput(value="4"),
    )

    score = metric.score(ParsedOutput(value="4"), case, ctx)

    assert isinstance(score, Score)
    assert score.value == 0.0


def test_f1_metric_returns_one_for_both_empty_sequences() -> None:
    metric = cast(PureMetric, load("builtin/f1"))
    case = Case(case_id="case-1", input={"question": "2+2"}, expected_output=None)
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_output=ParsedOutput(value=None),
    )

    score = metric.score(ParsedOutput(value=None), case, ctx)

    assert isinstance(score, Score)
    assert score.value == 1.0


def test_bleu_metric_returns_zero_for_empty_prediction() -> None:
    metric = cast(PureMetric, load("builtin/bleu"))
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="the quick brown fox",
    )
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_output=ParsedOutput(value=None),
    )

    score = metric.score(ParsedOutput(value=None), case, ctx)

    assert isinstance(score, Score)
    assert score.value == 0.0
