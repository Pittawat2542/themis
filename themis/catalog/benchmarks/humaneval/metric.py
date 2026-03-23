"""Execution metric for HumanEval and HumanEval+."""

from __future__ import annotations

import json
import pprint
import re

from themis.records import MetricScore
from themis.types.json_validation import validate_json_dict

from .dataset import _json_loads_unbounded
from ...runtime.code_execution import SandboxExecutor, _default_executor

_PYTHON_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*(?P<code>.+?)\s*```",
    re.IGNORECASE | re.DOTALL,
)


class HumanEvalExecutionMetric:
    def __init__(
        self,
        *,
        metric_id: str,
        executor: SandboxExecutor | None = None,
    ) -> None:
        self._metric_id = metric_id
        self._executor = executor or _default_executor()

    def score(self, trial, candidate, context):
        del trial
        raw_text = ""
        inference = getattr(candidate, "inference", None)
        if inference is not None and getattr(inference, "raw_text", None) is not None:
            raw_text = str(inference.raw_text)
        code = _normalize_candidate_code(raw_text)
        details: dict[str, object] = {
            "base_status": "fail",
            "plus_status": None,
            "base_passed": 0,
            "base_total": len(_test_cases(context.get("official_tests"))),
            "plus_passed": 0,
            "plus_total": len(_test_cases(context.get("plus_tests"))),
        }
        if not code.strip():
            details["error"] = "empty_candidate"
            return MetricScore(
                metric_id=self._metric_id,
                value=0.0,
                details=validate_json_dict(details, label="humaneval score details"),
                error="Candidate output did not contain runnable Python code.",
            )
        if str(context.get("language", "")).strip().lower() != "python":
            raise ValueError("HumanEval requires language='python'.")
        if str(context.get("execution_mode", "")).strip().lower() != "function":
            raise ValueError("HumanEval requires execution_mode='function'.")
        function_name = str(context.get("function_name", "")).strip()
        if not function_name:
            raise ValueError("HumanEval requires function_name.")

        harness = _build_humaneval_harness(
            candidate_code=code,
            function_name=function_name,
            score_variant=str(context.get("score_variant", "base") or "base"),
            official_tests=_test_cases(context.get("official_tests")),
            plus_tests=_test_cases(context.get("plus_tests")),
            base_expected=list(context.get("base_expected", [])),
            plus_expected=list(context.get("plus_expected", [])),
            base_time_limits=_time_limits(context.get("base_time_limits")),
            plus_time_limits=_time_limits(context.get("plus_time_limits")),
            atol=float(context.get("atol", 0.0) or 0.0),
        )
        timeout_seconds = (
            1.0
            + sum(_time_limits(context.get("base_time_limits")))
            + sum(_time_limits(context.get("plus_time_limits")))
        )
        execution = self._executor.execute(
            code=harness,
            language="python",
            timeout_seconds=timeout_seconds,
        )
        if not execution.ok:
            details["error"] = execution.message or execution.stderr
            return MetricScore(
                metric_id=self._metric_id,
                value=0.0,
                details=validate_json_dict(details, label="humaneval score details"),
                error="HumanEval execution failed.",
            )
        try:
            payload = json.loads(execution.stdout)
        except json.JSONDecodeError as exc:
            details["error"] = f"invalid_harness_output: {exc}"
            return MetricScore(
                metric_id=self._metric_id,
                value=0.0,
                details=validate_json_dict(details, label="humaneval score details"),
                error="HumanEval harness did not return valid JSON.",
            )
        details.update(
            {
                "base_status": payload.get("base_status", "fail"),
                "plus_status": payload.get("plus_status"),
                "base_passed": int(payload.get("base_passed", 0) or 0),
                "base_total": int(
                    payload.get("base_total", details["base_total"]) or 0
                ),
                "plus_passed": int(payload.get("plus_passed", 0) or 0),
                "plus_total": int(
                    payload.get("plus_total", details["plus_total"]) or 0
                ),
            }
        )
        score_value = 1.0 if details["base_status"] == "pass" else 0.0
        if str(context.get("score_variant", "base")) == "plus":
            score_value = (
                1.0
                if details["base_status"] == "pass" and details["plus_status"] == "pass"
                else 0.0
            )
        return MetricScore(
            metric_id=self._metric_id,
            value=score_value,
            details=validate_json_dict(details, label="humaneval score details"),
            error=None if score_value == 1.0 else "One or more HumanEval tests failed.",
        )


def _normalize_candidate_code(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    matches = list(_PYTHON_FENCE_PATTERN.finditer(text))
    if matches:
        return matches[-1].group("code").strip()
    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("def ", "class ", "import ", "from ")):
            return "\n".join(lines[index:]).strip()
    return stripped


def _test_cases(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {"input": str(item.get("input", "")), "output": str(item.get("output", ""))}
        for item in value
        if isinstance(item, dict)
    ]


def _time_limits(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    limits: list[float] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            limits.append(float(item))
    return limits


def _build_humaneval_harness(
    *,
    candidate_code: str,
    function_name: str,
    score_variant: str,
    official_tests: list[dict[str, str]],
    plus_tests: list[dict[str, str]],
    base_expected: list[object],
    plus_expected: list[object],
    base_time_limits: list[float],
    plus_time_limits: list[float],
    atol: float,
) -> str:
    base_inputs = [_json_loads_unbounded(test["input"]) for test in official_tests]
    plus_inputs = [_json_loads_unbounded(test["input"]) for test in plus_tests]
    return (
        "import copy\n"
        "import json\n"
        "import math\n"
        "import time\n\n"
        f"BASE_INPUTS = {_python_literal(base_inputs)}\n"
        f"PLUS_INPUTS = {_python_literal(plus_inputs)}\n"
        f"BASE_EXPECTED = {_python_literal(base_expected)}\n"
        f"PLUS_EXPECTED = {_python_literal(plus_expected)}\n"
        f"BASE_TIME_LIMITS = {_python_literal(base_time_limits)}\n"
        f"PLUS_TIME_LIMITS = {_python_literal(plus_time_limits)}\n"
        f"ATOL = {_python_literal(atol)}\n"
        f"SCORE_VARIANT = {_python_literal(score_variant)}\n\n"
        + _HARNESS_SUPPORT
        + "\n"
        + candidate_code
        + "\n\n"
        + f"FUNCTION_NAME = {function_name!r}\n"
        + _HARNESS_MAIN
    )


def _python_literal(value: object) -> str:
    return pprint.pformat(value, width=100, sort_dicts=False)


_HARNESS_SUPPORT = """
def _resolve_callable(function_name):
    target = globals().get(function_name)
    if callable(target):
        return target
    solution_cls = globals().get("Solution")
    if isinstance(solution_cls, type) and hasattr(solution_cls, function_name):
        return getattr(solution_cls(), function_name)
    raise RuntimeError(f"Missing callable {function_name!r} in candidate code.")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches(actual, expected, atol):
    if _is_number(actual) and _is_number(expected):
        if actual == expected:
            return True
        return math.isclose(actual, expected, rel_tol=1e-07, abs_tol=max(atol, 1e-6 if isinstance(expected, float) else atol))
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_matches(a, e, atol) for a, e in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return all(_matches(a, e, atol) for a, e in zip(actual, expected))
    return actual == expected


def _poly(coefficients, value):
    total = 0.0
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        total += coefficient * (value ** (degree - index))
    return total


def _find_zero_matches(inp, actual, atol):
    coefficients = inp[0] if len(inp) == 1 and isinstance(inp[0], list) else inp
    if not _is_number(actual):
        return False
    return abs(_poly(coefficients, actual)) <= atol


def _run_group(function_name, inputs, expected, time_limits, use_find_zero):
    target = _resolve_callable(function_name)
    passed = 0
    failures = []
    for index, raw_input in enumerate(inputs):
        args = copy.deepcopy(raw_input)
        if not isinstance(args, list):
            args = [args]
        start = time.perf_counter()
        try:
            actual = target(*args)
        except Exception as exc:
            failures.append({"test_index": index + 1, "reason": "exception", "message": repr(exc)})
            continue
        elapsed = time.perf_counter() - start
        if index < len(time_limits) and elapsed > time_limits[index]:
            failures.append({"test_index": index + 1, "reason": "timeout", "elapsed": elapsed, "limit": time_limits[index]})
            continue
        expected_value = expected[index]
        ok = _find_zero_matches(args, actual, ATOL) if use_find_zero else _matches(actual, expected_value, ATOL)
        if ok:
            passed += 1
            continue
        failures.append({"test_index": index + 1, "reason": "wrong_answer", "expected": expected_value, "actual": actual})
    status = "pass" if passed == len(inputs) else "fail"
    return {"status": status, "passed": passed, "total": len(inputs), "failures": failures[:3]}
"""


_HARNESS_MAIN = """
base_result = _run_group(
    FUNCTION_NAME,
    BASE_INPUTS,
    BASE_EXPECTED,
    BASE_TIME_LIMITS,
    FUNCTION_NAME == "find_zero",
)
plus_result = None
if SCORE_VARIANT == "plus":
    plus_result = _run_group(
        FUNCTION_NAME,
        PLUS_INPUTS,
        PLUS_EXPECTED,
        PLUS_TIME_LIMITS,
        FUNCTION_NAME == "find_zero",
    )
print(
    json.dumps(
        {
            "base_status": base_result["status"],
            "plus_status": None if plus_result is None else plus_result["status"],
            "base_passed": base_result["passed"],
            "base_total": base_result["total"],
            "plus_passed": 0 if plus_result is None else plus_result["passed"],
            "plus_total": 0 if plus_result is None else plus_result["total"],
        }
    )
)
"""
