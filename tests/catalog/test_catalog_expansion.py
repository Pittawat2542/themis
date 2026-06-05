from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.loaders import BenchmarkSourceRequest
from themis.catalog.registry import list_component_ids
from themis.core.base import JSONValue
from themis.core.builtins import resolve_parser_component
from themis.core.config import TargetSpec
from themis.core.contexts import ParseContext, ScoreContext
from themis.core.models import ParsedOutput, ReducedCandidate, MetricResult
from themis.core.protocols import Parser, PureMetric


def test_benchmark_materialize_dataset_uses_fixture_loader_and_real_cases() -> None:
    benchmark = cast(object, load("mmlu_pro"))
    materialize_dataset = getattr(benchmark, "materialize_dataset")
    captured: list[BenchmarkSourceRequest] = []

    def loader(request: BenchmarkSourceRequest) -> list[dict[str, object]]:
        captured.append(request)
        return [
            {
                "question": "Which planet is known as the Red Planet?",
                "options": ["Venus", "Mars", "Jupiter", "Mercury"],
                "answer": "B",
                "answer_index": 1,
                "category": "astronomy",
                "src": "fixture",
            }
        ]

    dataset = materialize_dataset(loader=loader)

    assert dataset.dataset_id == "TIGER-Lab/MMLU-Pro"
    assert dataset.revision == "test"
    assert len(dataset.cases) == 1
    assert captured == [
        BenchmarkSourceRequest(
            dataset_id="TIGER-Lab/MMLU-Pro",
            split="test",
            source_kind="huggingface_dataset",
        )
    ]
    assert dataset.cases[0].input == (
        "Question:\nWhich planet is known as the Red Planet?\n\n"
        "Options:\nA. Venus\nB. Mars\nC. Jupiter\nD. Mercury\n\n"
        "Return the best option letter only."
    )
    assert dataset.cases[0].expected_output == {"choice": "B"}
    assert dataset.cases[0].metadata["category"] == "astronomy"


def test_catalog_exposes_reusable_parser_and_metric_components() -> None:
    parser = cast(Parser, load("builtin/choice_letter"))
    metric = cast(PureMetric, load("builtin/choice_accuracy"))

    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1",
            final_output="The best answer is (B).",
        ),
        ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1"),
    )

    score = metric.score(
        parsed,
        benchmark_case(
            input_value="Question:\n2+2?\n\nOptions:\nA. 3\nB. 4",
            expected_output={"choice": "B"},
        ),
        ScoreContext(
            run_id="run-1",
            case=benchmark_case(
                input_value="Question:\n2+2?\n\nOptions:\nA. 3\nB. 4",
                expected_output={"choice": "B"},
            ),
            parsed_views={"default": parsed},
        ),
    )

    assert parsed == ParsedOutput(value="B", format="choice_letter")
    assert isinstance(score, MetricResult)
    assert score.value == 1.0
    assert "builtin/choice_letter" in list_component_ids(kind="parser")
    assert "builtin/choice_accuracy" in list_component_ids(kind="metric")


def test_catalog_math_answer_and_metric_are_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMathVerify:
        @staticmethod
        def parse(value: str) -> str:
            return value.strip().strip("{}")

        @staticmethod
        def verify(gold: str, candidate: str) -> bool:
            return gold == candidate

    import themis.catalog.builtins.metrics as builtin_metrics

    monkeypatch.setattr(builtin_metrics, "_import_math_verify", lambda: _FakeMathVerify)

    parser = cast(Parser, load("builtin/math_answer"))
    metric = cast(PureMetric, load("builtin/math_equivalence"))
    case = benchmark_case(
        input_value="Solve 2 + 2.",
        expected_output={"answer": "4"},
    )
    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1",
            final_output="The answer is \\boxed{4}.",
        ),
        ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1"),
    )

    score = metric.score(
        parsed,
        case,
        ScoreContext(run_id="run-1", case=case, parsed_views={"default": parsed}),
    )

    assert parsed == ParsedOutput(value="4", format="math_answer")
    assert isinstance(score, MetricResult)
    assert score.value == 1.0


def test_catalog_code_execution_metric_is_reusable() -> None:
    from themis.catalog.builtins.code_execution import (
        CodeforcesExecutionMetric,
        SandboxExecutionResult,
    )

    class _FakeExecutor:
        def execute(
            self,
            *,
            code: str,
            language: str,
            stdin: str = "",
            files: dict[str, str] | None = None,
            args: list[str] | None = None,
            timeout_seconds: float | None = None,
            memory_limit_mb: float | None = None,
        ) -> SandboxExecutionResult:
            del code, language, files, args, timeout_seconds, memory_limit_mb
            return SandboxExecutionResult(
                stdout="4\n" if stdin == "6 6 4\n" else "",
                stderr="",
                return_code=0,
                status="ok",
            )

    metric = CodeforcesExecutionMetric(executor=_FakeExecutor())
    case = benchmark_case(
        input_value="Write a Python program that solves the problem.",
        expected_output={
            "language": "python",
            "execution_mode": "stdio",
            "official_tests": [{"input": "6 6 4\n", "output": "4\n"}],
        },
    )
    parsed = ParsedOutput(value="print(4)")

    score = metric.score(
        parsed,
        case,
        ScoreContext(run_id="run-1", case=case, parsed_views={"default": parsed}),
    )

    assert isinstance(score, MetricResult)
    assert score.metric_id == "builtin/codeforces_pass_rate"
    assert score.value == 1.0


def test_regex_parser_extracts_whole_numeric_and_named_groups() -> None:
    parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/regex",
                kwargs={"pattern": r"answer:\s*(?P<answer>[A-Z])", "group": "answer"},
            )
        ),
    )
    numeric_parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/regex",
                kwargs={"pattern": r"score=(\d+)", "group": 1},
            )
        ),
    )
    whole_match_parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/regex",
                kwargs={"pattern": r"final:\s*\d+"},
            )
        ),
    )
    ctx = ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1")

    named = parser.parse(
        ReducedCandidate(candidate_id="candidate-1", final_output="answer: B"), ctx
    )
    numeric = numeric_parser.parse(
        ReducedCandidate(candidate_id="candidate-1", final_output="score=42"), ctx
    )
    whole = whole_match_parser.parse(
        ReducedCandidate(candidate_id="candidate-1", final_output="final: 42"), ctx
    )

    assert named == ParsedOutput(
        value="B", format="regex", metadata={"group": "answer"}
    )
    assert numeric == ParsedOutput(value="42", format="regex", metadata={"group": 1})
    assert whole == ParsedOutput(value="final: 42", format="regex")


def test_regex_parser_supports_flags_and_reports_failures() -> None:
    parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/regex",
                kwargs={
                    "pattern": r"^answer:\s*(.+)$",
                    "group": 1,
                    "ignore_case": True,
                    "multiline": True,
                },
            )
        ),
    )
    ctx = ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1")

    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1", final_output="notes\nANSWER: Correct"
        ),
        ctx,
    )

    assert parsed == ParsedOutput(
        value="Correct", format="regex", metadata={"group": 1}
    )
    with pytest.raises(ValueError, match="Regex did not match"):
        parser.parse(
            ReducedCandidate(candidate_id="candidate-1", final_output="no answer"),
            ctx,
        )
    with pytest.raises(ValueError, match="Regex group is not available"):
        cast(
            Parser,
            resolve_parser_component(
                TargetSpec(
                    target="builtin/regex",
                    kwargs={"pattern": r"answer:\s*(.+)", "group": "missing"},
                )
            ),
        ).parse(
            ReducedCandidate(candidate_id="candidate-1", final_output="answer: A"),
            ctx,
        )


def test_schema_parser_validates_structured_values_and_extracts_path() -> None:
    parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/schema",
                kwargs={
                    "schema": {
                        "type": "object",
                        "required": ["answer", "steps"],
                        "properties": {
                            "answer": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                    },
                    "path": "answer",
                },
            )
        ),
    )
    ctx = ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1")

    parsed_from_json = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1",
            final_output='{"answer": "4", "steps": [1, 2]}',
        ),
        ctx,
    )
    parsed_from_dict = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1",
            final_output={"answer": "5", "steps": [3]},
        ),
        ctx,
    )

    assert parsed_from_json == ParsedOutput(
        value="4", format="schema", metadata={"path": "answer"}
    )
    assert parsed_from_dict == ParsedOutput(
        value="5", format="schema", metadata={"path": "answer"}
    )


def test_schema_parser_reports_json_shape_and_path_failures() -> None:
    parser = cast(
        Parser,
        resolve_parser_component(
            TargetSpec(
                target="builtin/schema",
                kwargs={
                    "schema": {
                        "type": "object",
                        "required": ["answer"],
                        "properties": {"answer": {"type": "string"}},
                    },
                    "path": "answer.text",
                },
            )
        ),
    )
    ctx = ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1")

    with pytest.raises(ValueError, match="Invalid JSON"):
        parser.parse(
            ReducedCandidate(candidate_id="candidate-1", final_output="{bad json"),
            ctx,
        )
    with pytest.raises(ValueError, match="Missing required field"):
        parser.parse(
            ReducedCandidate(candidate_id="candidate-1", final_output={}),
            ctx,
        )
    with pytest.raises(ValueError, match="Expected string"):
        parser.parse(
            ReducedCandidate(candidate_id="candidate-1", final_output={"answer": 4}),
            ctx,
        )
    with pytest.raises(ValueError, match="Schema path is not available"):
        parser.parse(
            ReducedCandidate(candidate_id="candidate-1", final_output={"answer": "4"}),
            ctx,
        )


def test_rouge_metrics_are_manifest_backed_and_score_overlap() -> None:
    rouge1 = cast(PureMetric, load("builtin/rouge1"))
    rouge2 = cast(PureMetric, load("builtin/rouge2"))
    rouge_l = cast(PureMetric, load("builtin/rouge_l"))
    case = benchmark_case(
        input_value="Summarize.",
        expected_output="the quick brown fox",
    )
    ctx = ScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": ParsedOutput(value="the quick fox")},
    )

    rouge1_score = rouge1.score(ParsedOutput(value="the quick quick fox"), case, ctx)
    rouge2_score = rouge2.score(ParsedOutput(value="the quick fox"), case, ctx)
    rouge_l_score = rouge_l.score(ParsedOutput(value="quick brown"), case, ctx)

    assert "builtin/regex" in list_component_ids(kind="parser")
    assert "builtin/schema" in list_component_ids(kind="parser")
    assert "builtin/rouge1" in list_component_ids(kind="metric")
    assert "builtin/rouge2" in list_component_ids(kind="metric")
    assert "builtin/rouge_l" in list_component_ids(kind="metric")
    assert isinstance(rouge1_score, MetricResult)
    assert rouge1_score.value == pytest.approx(0.75)
    assert rouge1_score.dimensions == pytest.approx(
        {"precision": 0.75, "recall": 0.75, "f1": 0.75}
    )
    assert isinstance(rouge2_score, MetricResult)
    assert rouge2_score.value == pytest.approx(0.4)
    assert isinstance(rouge_l_score, MetricResult)
    assert rouge_l_score.value == pytest.approx(2 / 3)


def benchmark_case(*, input_value: JSONValue, expected_output: JSONValue):
    from themis.core.models import Case

    return Case(case_id="case-1", input=input_value, expected_output=expected_output)
