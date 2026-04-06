from __future__ import annotations

from typing import cast

from themis.catalog import load
from themis.catalog.builtins.code_execution import (
    HumanEvalExecutionMetric,
    SandboxExecutionResult,
)
from themis.catalog.builtins.parsers import CodeTextParser
from themis.catalog.benchmarks import BenchmarkDefinition
from themis.core.base import JSONValue
from themis.core.contexts import ParseContext, ScoreContext
from themis.core.models import ParsedOutput, ReducedCandidate, Score


def test_humaneval_benchmarks_use_code_execution_wiring() -> None:
    humaneval = cast(BenchmarkDefinition, load("humaneval:v0.1.0"))
    humaneval_plus = cast(BenchmarkDefinition, load("humaneval_plus:noextreme"))

    assert humaneval.metric_ids == ["builtin/humaneval_pass_rate"]
    assert humaneval.parser_ids == ["builtin/code_text"]
    assert humaneval.support_tier == "ready"
    assert humaneval_plus.metric_ids == ["builtin/humaneval_pass_rate"]
    assert humaneval_plus.parser_ids == ["builtin/code_text"]
    assert humaneval_plus.support_tier == "ready"


def test_humaneval_materialization_includes_reference_solution_for_demo_runs() -> None:
    humaneval = cast(BenchmarkDefinition, load("humaneval:v0.1.0"))

    dataset = humaneval.materialize_dataset()
    payload = dataset.cases[0].expected_output

    assert isinstance(payload, dict)
    assert payload["execution_mode"] == "function"
    assert payload["function_name"] == "square"
    assert "reference_solution" in payload
    assert "solution" in payload
    assert payload["solution"] == payload["reference_solution"]
    assert "def square" in str(payload["reference_solution"])


def test_humaneval_execution_metric_scores_candidate_against_reference_solution() -> (
    None
):
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
            del language, stdin, files, args, timeout_seconds, memory_limit_mb
            if "return a + b" in code and "[1, 2]" in code:
                return SandboxExecutionResult(
                    stdout="3\n", stderr="", return_code=0, status="ok"
                )
            if "return a + b" in code and "[4, 5]" in code:
                return SandboxExecutionResult(
                    stdout="9\n", stderr="", return_code=0, status="ok"
                )
            return SandboxExecutionResult(
                stdout="0\n", stderr="", return_code=0, status="ok"
            )

    metric = HumanEvalExecutionMetric(executor=_FakeExecutor())
    parser = CodeTextParser()
    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="candidate-1",
            final_output={"solution": "def add(a, b):\n    return a + b\n"},
        ),
        ParseContext(run_id="run-1", case_id="case-1", candidate_id="candidate-1"),
    )
    case_obj = benchmark_case(
        expected_output={
            "language": "python",
            "execution_mode": "function",
            "function_name": "add",
            "official_tests": [{"input": "[1, 2]"}, {"input": "[4, 5]"}],
            "reference_solution": "def add(a, b):\n    return a + b\n",
            "solution": "def add(a, b):\n    return a + b\n",
            "score_variant": "base",
        }
    )
    score = metric.score(
        parsed,
        case_obj,
        ScoreContext(
            run_id="run-1",
            case=case_obj,
            parsed_output=parsed,
        ),
    )

    assert parsed == ParsedOutput(
        value="def add(a, b):\n    return a + b",
        format="code",
    )
    assert isinstance(score, Score)
    assert score.metric_id == "builtin/humaneval_pass_rate"
    assert score.value == 1.0


def test_humaneval_execution_metric_caches_reference_solution_results() -> None:
    class _CountingExecutor:
        def __init__(self) -> None:
            self.reference_runs: list[str] = []

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
            del language, stdin, files, args, timeout_seconds, memory_limit_mb
            if "return a + b" in code:
                self.reference_runs.append(code)
            return SandboxExecutionResult(
                stdout="3\n", stderr="", return_code=0, status="ok"
            )

    executor = _CountingExecutor()
    metric = HumanEvalExecutionMetric(executor=executor)
    case_obj = benchmark_case(
        expected_output={
            "language": "python",
            "execution_mode": "function",
            "function_name": "add",
            "official_tests": [{"input": "[1, 2]"}, {"input": "[1, 2]"}],
            "reference_solution": "def add(a, b):\n    return a + b\n",
            "solution": "def add(a, b):\n    return a + b\n",
            "score_variant": "base",
        }
    )

    score = metric.score(
        ParsedOutput(value="def add(a, b):\n    return 1 + 2", format="code"),
        case_obj,
        ScoreContext(
            run_id="run-1",
            case=case_obj,
            parsed_output=ParsedOutput(
                value="def add(a, b):\n    return 1 + 2",
                format="code",
            ),
        ),
    )

    assert isinstance(score, Score)
    assert score.value == 1.0
    assert len(executor.reference_runs) == 1


def benchmark_case(*, expected_output: JSONValue):
    from themis.core.models import Case

    return Case(
        case_id="case-1",
        input="Write code.",
        expected_output=expected_output,
    )
