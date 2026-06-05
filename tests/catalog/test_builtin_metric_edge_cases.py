from __future__ import annotations

from typing import cast

from themis.catalog import load
from themis.core.contexts import ScoreContext
from themis.core.models import Case, ParsedOutput, MetricResult
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
        parsed_views={"default": ParsedOutput(value="4")},
    )

    score = metric.score(ParsedOutput(value="4"), case, ctx)

    assert isinstance(score, MetricResult)
    assert score.value == 0.0


def test_f1_metric_returns_one_for_both_empty_sequences() -> None:
    metric = cast(PureMetric, load("builtin/f1"))
    case = Case(case_id="case-1", input={"question": "2+2"}, expected_output=None)
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": ParsedOutput(value=None)},
    )

    score = metric.score(ParsedOutput(value=None), case, ctx)

    assert isinstance(score, MetricResult)
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
        parsed_views={"default": ParsedOutput(value=None)},
    )

    score = metric.score(ParsedOutput(value=None), case, ctx)

    assert isinstance(score, MetricResult)
    assert score.value == 0.0


def test_rouge_metrics_return_one_when_prediction_and_expected_are_empty() -> None:
    for metric_id in ("builtin/rouge1", "builtin/rouge2", "builtin/rouge_l"):
        metric = cast(PureMetric, load(metric_id))
        case = Case(case_id="case-1", input="summarize", expected_output="")
        ctx = ScoreContext(
            run_id="run-1",
            case=case,
            parsed_views={"default": ParsedOutput(value="")},
        )

        score = metric.score(ParsedOutput(value=""), case, ctx)

        assert isinstance(score, MetricResult)
        assert score.value == 1.0
        assert score.dimensions == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_rouge_metrics_return_zero_when_only_one_side_is_empty() -> None:
    for metric_id in ("builtin/rouge1", "builtin/rouge2", "builtin/rouge_l"):
        metric = cast(PureMetric, load(metric_id))
        case = Case(case_id="case-1", input="summarize", expected_output="not empty")
        ctx = ScoreContext(
            run_id="run-1",
            case=case,
            parsed_views={"default": ParsedOutput(value="")},
        )

        score = metric.score(ParsedOutput(value=""), case, ctx)

        assert isinstance(score, MetricResult)
        assert score.value == 0.0
        assert score.dimensions == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
