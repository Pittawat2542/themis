from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load
from themis.core.contexts import GenerationContext, ParseContext, ScoreContext
from themis.core.models import Case, ParsedOutput, ReducedCandidate, MetricResult
from themis.core.protocols import Generator, Parser, PureMetric


@pytest.mark.asyncio
async def test_catalog_builtin_generator_and_parser_execute_through_manifest() -> None:
    generator = cast(Generator, load("builtin/demo_generator"))
    parser = cast(Parser, load("builtin/json_identity"))
    case = Case(
        case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
    )

    generated = await generator.generate(
        case, GenerationContext(run_id="run-1", case_id="case-1", seed=7)
    )
    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="case-1-reduced",
            source_candidate_ids=[generated.candidate_id],
            final_output=generated.final_output,
        ),
        ParseContext(run_id="run-1", case_id="case-1", candidate_id="case-1-reduced"),
    )

    assert generated.final_output == {"answer": "4"}
    assert parsed == ParsedOutput(value={"answer": "4"}, format="json")


def test_catalog_builtin_pure_metrics_score_expected_values() -> None:
    exact_match = cast(PureMetric, load("builtin/exact_match"))
    f1 = cast(PureMetric, load("builtin/f1"))
    bleu = cast(PureMetric, load("builtin/bleu"))
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="the quick brown fox",
    )
    exact_ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": ParsedOutput(value="the quick brown fox")},
    )

    exact_score = exact_match.score(
        ParsedOutput(value="the quick brown fox"), case, exact_ctx
    )
    f1_score = f1.score(ParsedOutput(value="quick fox"), case, exact_ctx)
    bleu_score = bleu.score(ParsedOutput(value="the quick fox"), case, exact_ctx)

    assert isinstance(exact_score, MetricResult)
    assert isinstance(f1_score, MetricResult)
    assert isinstance(bleu_score, MetricResult)
    assert exact_score.value == 1.0
    assert f1_score.value == 2 / 3
    assert bleu_score.value == 1.0


def test_catalog_exposes_semantic_similarity_metric() -> None:
    metric = cast(PureMetric, load("builtin/semantic_similarity"))
    case = Case(
        case_id="case-1",
        input="answer",
        expected_output="the quick brown fox",
    )
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": ParsedOutput(value="quick brown")},
    )

    score = metric.score(ParsedOutput(value="quick brown"), case, ctx)

    assert isinstance(score, MetricResult)
    assert score.metric_id == "builtin/semantic_similarity"
    assert score.value == 0.707106781187
    assert score.dimensions["overlap"] == 2.0
    assert score.metadata["method"] == "token_cosine"


def test_catalog_exposes_calibration_and_agreement_metrics() -> None:
    calibration = cast(PureMetric, load("builtin/confidence_calibration"))
    agreement = cast(PureMetric, load("builtin/label_agreement"))
    case = Case(case_id="case-1", input="classify", expected_output="yes")
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": ParsedOutput(value="yes", confidence=0.8)},
    )

    calibration_score = calibration.score(
        ParsedOutput(value="yes", confidence=0.8), case, ctx
    )
    agreement_score = agreement.score(
        ParsedOutput(value=["safe", "safe", "unsafe"]), case, ctx
    )

    assert isinstance(calibration_score, MetricResult)
    assert calibration_score.result_type == "calibration"
    assert calibration_score.value == 0.2
    assert isinstance(agreement_score, MetricResult)
    assert agreement_score.result_type == "agreement"
    assert agreement_score.value == 2 / 3
    assert agreement_score.labels == {"majority_label": "safe"}


def test_catalog_exposes_robustness_baseline_ablation_and_budget_components() -> None:
    lowercase = load("builtin/lowercase_transform")
    baseline_pack = load("builtin/default_baseline_pack")
    ablation = load("builtin/parser_ablation_template")
    budget_sweep = load("builtin/token_budget_sweep")

    assert getattr(lowercase, "apply")("HeLLo") == "hello"
    assert "builtin/demo_generator" in getattr(baseline_pack, "component_ids")
    assert getattr(ablation, "variants")("builtin/json_identity") == [
        "builtin/json_identity::disabled",
        "builtin/json_identity::fallback_only",
    ]
    assert getattr(budget_sweep, "budgets") == [128, 512, 2048]
