from __future__ import annotations

from typing import cast

import pytest

from themis.catalog import load
from themis.catalog.loaders import BenchmarkSourceRequest
from themis.catalog.registry import list_component_ids
from themis.core.base import JSONValue
from themis.core.contexts import ParseContext, ScoreContext
from themis.core.models import ParsedOutput, ReducedCandidate, Score
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
            parsed_output=parsed,
        ),
    )

    assert parsed == ParsedOutput(value="B", format="choice_letter")
    assert isinstance(score, Score)
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
        ScoreContext(run_id="run-1", case=case, parsed_output=parsed),
    )

    assert parsed == ParsedOutput(value="4", format="math_answer")
    assert isinstance(score, Score)
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
        ScoreContext(run_id="run-1", case=case, parsed_output=parsed),
    )

    assert isinstance(score, Score)
    assert score.metric_id == "builtin/codeforces_pass_rate"
    assert score.value == 1.0


def benchmark_case(*, input_value: JSONValue, expected_output: JSONValue):
    from themis.core.models import Case

    return Case(case_id="case-1", input=input_value, expected_output=expected_output)
