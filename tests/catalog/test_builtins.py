from __future__ import annotations

import pytest

from themis.catalog import load
from themis.core.contexts import GenerateContext, ParseContext, ScoreContext
from themis.core.models import Case, ParsedOutput, ReducedCandidate


@pytest.mark.asyncio
async def test_catalog_builtin_generator_and_parser_execute_through_manifest() -> None:
    generator = load("builtin/demo_generator")
    parser = load("builtin/json_identity")
    case = Case(
        case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
    )

    generated = await generator.generate(
        case, GenerateContext(run_id="run-1", case_id="case-1", seed=7)
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
    exact_match = load("builtin/exact_match")
    f1 = load("builtin/f1")
    bleu = load("builtin/bleu")
    case = Case(
        case_id="case-1",
        input={"question": "2+2"},
        expected_output="the quick brown fox",
    )
    exact_ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_output=ParsedOutput(value="the quick brown fox"),
    )

    exact_score = exact_match.score(
        ParsedOutput(value="the quick brown fox"), case, exact_ctx
    )
    f1_score = f1.score(ParsedOutput(value="quick fox"), case, exact_ctx)
    bleu_score = bleu.score(ParsedOutput(value="the quick fox"), case, exact_ctx)

    assert exact_score.value == 1.0
    assert f1_score.value == 2 / 3
    assert bleu_score.value == 1.0
