"""Reusable code-execution metrics for catalog benchmarks."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib import request

from themis.core.base import JSONValue
from themis.core.contexts import ScoreContext
from themis.core.models import Case, ParsedOutput, Score


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    status: str
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.return_code == 0 and self.status == "ok"


class SandboxExecutor(Protocol):
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
    ) -> SandboxExecutionResult: ...


class PistonSandboxExecutor:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        urlopen=request.urlopen,
    ) -> None:
        self._base_url = (
            base_url or os.getenv("THEMIS_CODE_PISTON_URL") or "http://localhost:2000"
        ).rstrip("/")
        self._urlopen = urlopen

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
        del timeout_seconds, memory_limit_mb
        piston_language, piston_version = _resolve_piston_runtime(language)
        payload = {
            "language": piston_language,
            "version": piston_version,
            "files": [
                {"name": name, "content": content}
                for name, content in _ordered_files(code, language, files).items()
            ],
            "stdin": stdin,
            "args": list(args or []),
        }
        req = request.Request(
            f"{self._base_url}/api/v2/execute",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("Piston sandbox returned a non-object response.")
        run_payload = body.get("run") or {}
        compile_payload = body.get("compile") or {}
        stderr = str(compile_payload.get("stderr", "")) + str(
            run_payload.get("stderr", "")
        )
        status = (
            "ok"
            if int(run_payload.get("code", 1) or 0) == 0
            and not compile_payload.get("stderr")
            else "error"
        )
        return SandboxExecutionResult(
            stdout=str(run_payload.get("stdout", "")),
            stderr=stderr,
            return_code=int(run_payload.get("code", 1) or 0),
            status=status,
            message=str(body.get("message", "")).strip() or None,
        )


class SandboxFusionExecutor:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        urlopen=request.urlopen,
    ) -> None:
        self._base_url = (
            base_url
            or os.getenv("THEMIS_CODE_SANDBOX_FUSION_URL")
            or "http://localhost:8080"
        ).rstrip("/")
        self._urlopen = urlopen

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
        del timeout_seconds, memory_limit_mb
        payload: dict[str, object] = {
            "code": code,
            "language": _resolve_sandbox_fusion_language(language),
            "stdin": stdin,
            "args": list(args or []),
        }
        if files:
            payload["files"] = {
                str(name): base64.b64encode(str(content).encode("utf-8")).decode(
                    "ascii"
                )
                for name, content in files.items()
            }
        req = request.Request(
            f"{self._base_url}/run_code",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("SandboxFusion returned a non-object response.")
        run_payload = body.get("run_result") or {}
        status = (
            "ok"
            if str(body.get("status", "")) == "Success"
            and int(run_payload.get("return_code", 1) or 0) == 0
            else "error"
        )
        return SandboxExecutionResult(
            stdout=str(run_payload.get("stdout", "")),
            stderr=str(run_payload.get("stderr", "")),
            return_code=int(run_payload.get("return_code", 1) or 0),
            status=status,
            message=str(body.get("message", "")).strip() or None,
        )


class CodeExecutionMetric:
    version = "1.0"
    metric_family = "pure"

    def __init__(
        self,
        *,
        component_id: str,
        benchmark_name: str,
        supported_languages: set[str],
        supported_modes: set[str],
        executor: SandboxExecutor | None = None,
    ) -> None:
        self.component_id = component_id
        self._benchmark_name = benchmark_name
        self._supported_languages = {
            value.strip().lower() for value in supported_languages
        }
        self._supported_modes = {value.strip().lower() for value in supported_modes}
        self._executor = executor or SandboxFusionExecutor()

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        code = str(parsed.value).strip()
        payload = case.expected_output if isinstance(case.expected_output, dict) else {}
        language = str(payload.get("language", "")).strip().lower()
        execution_mode = str(payload.get("execution_mode", "stdio")).strip().lower()
        tests = _normalize_tests(payload.get("official_tests"))
        if execution_mode not in self._supported_modes:
            return _score(
                self.component_id, 0.0, {"reason": "unsupported_execution_mode"}
            )
        if language not in self._supported_languages:
            return _score(self.component_id, 0.0, {"reason": "unsupported_language"})
        if not code or not tests:
            return _score(self.component_id, 0.0, {"reason": "missing_code_or_tests"})

        passed = 0
        for test_case in tests:
            result = self._executor.execute(
                code=code,
                language=language,
                stdin=test_case["input"],
            )
            if result.ok and _normalize_output(result.stdout) == _normalize_output(
                test_case["output"]
            ):
                passed += 1
        total = len(tests)
        return _score(
            self.component_id,
            passed / total if total else 0.0,
            {
                "passed_tests": passed,
                "total_tests": total,
                "benchmark": self._benchmark_name,
            },
        )


class CodeforcesExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            component_id="builtin/codeforces_pass_rate",
            benchmark_name="codeforces",
            supported_languages={"python", "cpp", "cplusplus"},
            supported_modes={"stdio"},
            executor=executor,
        )


class AetherCodeExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            component_id="builtin/aethercode_pass_rate",
            benchmark_name="aethercode",
            supported_languages={"cpp", "cplusplus"},
            supported_modes={"stdio"},
            executor=executor,
        )


class LiveCodeBenchExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            component_id="builtin/livecodebench_pass_rate",
            benchmark_name="livecodebench",
            supported_languages={"python", "cpp", "cplusplus"},
            supported_modes={"stdio"},
            executor=executor,
        )


class HumanEvalExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            component_id="builtin/humaneval_pass_rate",
            benchmark_name="humaneval",
            supported_languages={"python"},
            supported_modes={"function"},
            executor=executor,
        )

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del ctx
        code = str(parsed.value).strip()
        payload = case.expected_output if isinstance(case.expected_output, dict) else {}
        language = str(payload.get("language", "")).strip().lower()
        execution_mode = str(payload.get("execution_mode", "function")).strip().lower()
        function_name = str(payload.get("function_name", "")).strip()
        reference_solution = str(payload.get("reference_solution", "")).strip()
        tests = _normalize_function_tests(payload.get("official_tests"))
        score_variant = str(payload.get("score_variant", "base")).strip().lower()
        if score_variant == "plus":
            plus_tests = _normalize_function_tests(payload.get("plus_tests"))
            if plus_tests:
                tests = plus_tests
        if execution_mode not in self._supported_modes:
            return _score(
                self.component_id, 0.0, {"reason": "unsupported_execution_mode"}
            )
        if language not in self._supported_languages:
            return _score(self.component_id, 0.0, {"reason": "unsupported_language"})
        if not code or not tests or not function_name or not reference_solution:
            return _score(
                self.component_id, 0.0, {"reason": "missing_code_or_reference"}
            )

        passed = 0
        reference_cache: dict[str, tuple[bool, str]] = {}
        for test_case in tests:
            candidate_result = self._executor.execute(
                code=_humaneval_wrapper(
                    solution=code,
                    function_name=function_name,
                    input_json=test_case["input"],
                ),
                language=language,
            )
            cache_key = json.dumps(test_case["input"], sort_keys=True)
            if cache_key not in reference_cache:
                reference_result = self._executor.execute(
                    code=_humaneval_wrapper(
                        solution=reference_solution,
                        function_name=function_name,
                        input_json=test_case["input"],
                    ),
                    language=language,
                )
                reference_cache[cache_key] = (
                    reference_result.ok,
                    _normalize_output(reference_result.stdout),
                )
            reference_ok, reference_stdout = reference_cache[cache_key]
            if (
                candidate_result.ok
                and reference_ok
                and _normalize_output(candidate_result.stdout) == reference_stdout
            ):
                passed += 1
        total = len(tests)
        return _score(
            self.component_id,
            passed / total if total else 0.0,
            {
                "passed_tests": passed,
                "total_tests": total,
                "benchmark": self._benchmark_name,
            },
        )


def _ordered_files(
    code: str,
    language: str,
    files: dict[str, str] | None,
) -> dict[str, str]:
    resolved = {str(name): str(content) for name, content in (files or {}).items()}
    main_name = "main.py" if language.strip().lower() == "python" else "main.cpp"
    if main_name in resolved:
        return resolved
    return {main_name: code, **resolved}


def _resolve_piston_runtime(language: str) -> tuple[str, str]:
    normalized = language.strip().lower()
    if normalized == "python":
        return ("python", "*")
    if normalized in {"cpp", "cplusplus"}:
        return ("c++", "*")
    raise ValueError(f"Unsupported code language '{language}'.")


def _resolve_sandbox_fusion_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized == "python":
        return "python"
    if normalized in {"cpp", "cplusplus"}:
        return "cpp"
    raise ValueError(f"Unsupported code language '{language}'.")


def _normalize_tests(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    tests: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        raw_input = entry.get("input")
        raw_output = entry.get("output")
        if isinstance(raw_input, str) and isinstance(raw_output, str):
            tests.append({"input": raw_input, "output": raw_output})
    return tests


def _normalize_function_tests(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    tests: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        raw_input = entry.get("input")
        if isinstance(raw_input, str) and raw_input:
            tests.append({"input": raw_input})
    return tests


def _humaneval_wrapper(
    *,
    solution: str,
    function_name: str,
    input_json: str,
) -> str:
    return "\n".join(
        [
            solution.rstrip(),
            "",
            "import json",
            "",
            f"_themis_args = json.loads({json.dumps(input_json)})",
            f"_themis_result = {function_name}(*_themis_args)",
            "print(json.dumps(_themis_result, sort_keys=True))",
            "",
        ]
    )


def _normalize_output(value: str) -> str:
    return value.replace("\r\n", "\n").strip()


def _score(
    metric_id: str,
    value: float,
    details: dict[str, object],
) -> Score:
    resolved: dict[str, JSONValue] = {}
    for key, item in details.items():
        if item is None or isinstance(item, (str, int, float, bool)):
            resolved[key] = item
        else:
            resolved[key] = str(item)
    return Score(metric_id=metric_id, value=value, details=resolved)
