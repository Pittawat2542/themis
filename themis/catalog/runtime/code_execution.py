"""Shared code-execution helpers for catalog benchmarks."""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import request

from themis.records import MetricScore
from themis.types.json_validation import validate_json_dict

from .._http import DEFAULT_HTTP_TIMEOUT_SECONDS, open_json_request


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
            "compile_timeout": _milliseconds(timeout_seconds, default_ms=10000),
            "run_timeout": _milliseconds(timeout_seconds, default_ms=3000),
            "compile_cpu_time": _milliseconds(timeout_seconds, default_ms=10000),
            "run_cpu_time": _milliseconds(timeout_seconds, default_ms=3000),
            "compile_memory_limit": _memory_bytes(memory_limit_mb),
            "run_memory_limit": _memory_bytes(memory_limit_mb),
        }
        req = request.Request(
            f"{self._base_url}/api/v2/execute",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = open_json_request(
            req,
            urlopen=self._urlopen,
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        )
        if not isinstance(body, dict):
            raise ValueError("Piston sandbox returned a non-object response.")
        if "message" in body and "run" not in body and "compile" not in body:
            return SandboxExecutionResult(
                stdout="",
                stderr="",
                return_code=1,
                status="error",
                message=str(body["message"]),
            )
        run_payload = body.get("run") or {}
        compile_payload = body.get("compile") or {}
        stderr = str(run_payload.get("stderr", ""))
        if compile_payload.get("stderr"):
            stderr = f"{compile_payload['stderr']}{stderr}"
        status = "ok"
        if run_payload.get("status") is not None or run_payload.get("code", 1) != 0:
            status = "error"
        return SandboxExecutionResult(
            stdout=str(run_payload.get("stdout", "")),
            stderr=stderr,
            return_code=int(run_payload.get("code", 1) or 0),
            status=status,
            message=_coalesce_message(run_payload, compile_payload),
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
            "code": _sandbox_fusion_code(
                code=code,
                language=language,
                files=files,
                args=args,
            ),
            "language": _resolve_sandbox_fusion_language(language),
            "stdin": stdin,
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
        body = open_json_request(
            req,
            urlopen=self._urlopen,
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        )
        if not isinstance(body, dict):
            raise ValueError("SandboxFusion returned a non-object response.")
        run_payload = body.get("run_result") or {}
        compile_payload = body.get("compile_result") or {}
        status_value = str(body.get("status", ""))
        run_status = str(run_payload.get("status", ""))
        status = (
            "ok"
            if status_value == "Success"
            and run_status == "Finished"
            and int(run_payload.get("return_code", 1) or 0) == 0
            else "error"
        )
        stderr = str(run_payload.get("stderr", ""))
        if compile_payload and compile_payload.get("stderr"):
            stderr = f"{compile_payload['stderr']}{stderr}"
        message = str(body.get("message", "")).strip() or None
        return SandboxExecutionResult(
            stdout=str(run_payload.get("stdout", "")),
            stderr=stderr,
            return_code=int(run_payload.get("return_code", 1) or 0),
            status=status,
            message=message,
        )


class CodeExecutionMetric:
    def __init__(
        self,
        *,
        metric_id: str,
        benchmark_name: str,
        supported_languages: set[str],
        supported_modes: set[str],
        executor: SandboxExecutor | None = None,
        checker_support_files: dict[str, str] | None = None,
    ) -> None:
        self._metric_id = metric_id
        self._benchmark_name = benchmark_name
        self._supported_languages = {
            value.strip().lower() for value in supported_languages
        }
        self._supported_modes = {value.strip().lower() for value in supported_modes}
        self._executor = executor or _default_executor()
        self._checker_support_files = {
            str(name): str(content)
            for name, content in (checker_support_files or {}).items()
        }

    def score(self, trial, candidate, context):
        del trial
        code = ""
        inference = getattr(candidate, "inference", None)
        if inference is not None and getattr(inference, "raw_text", None) is not None:
            code = str(inference.raw_text)
        language = str(context.get("language", "")).strip().lower()
        execution_mode = (
            str(context.get("execution_mode", context.get("input_mode", "")))
            .strip()
            .lower()
        )
        tests = _normalize_tests(context.get("official_tests"))
        checker = _normalize_checker(context.get("generated_checker"))
        checker_language = (
            str(
                context.get("checker_language", "python" if checker is not None else "")
            )
            .strip()
            .lower()
        )
        checker_support_files = self._resolve_checker_support_files(context)
        function_name = str(context.get("function_name", "")).strip() or None
        timeout_seconds = _coerce_float(context.get("time_limit"))
        memory_limit_mb = _coerce_float(context.get("memory_limit"))
        details: dict[str, object] = {
            "language": language,
            "input_mode": execution_mode,
            "total_tests": len(tests),
            "passed_tests": 0,
            "used_checker": bool(checker),
        }
        if execution_mode not in self._supported_modes:
            raise ValueError(
                f"{self._benchmark_name} only supports execution mode(s) "
                f"{sorted(self._supported_modes)}; got "
                f"input_mode='{execution_mode or '<missing>'}'."
            )
        if language not in self._supported_languages:
            supported = "', '".join(sorted(self._supported_languages))
            raise ValueError(
                f"{self._benchmark_name} requires a supported language "
                f"('{supported}'); got "
                f"'{language or '<missing>'}'."
            )
        if not code.strip():
            details["error"] = "empty_candidate"
            return MetricScore(
                metric_id=self._metric_id,
                value=0.0,
                details=details,
                error="Candidate output did not contain runnable code.",
            )
        if not tests:
            raise ValueError(
                f"{self._benchmark_name} requires official_tests for execution scoring."
            )

        if execution_mode == "function":
            if function_name is None:
                raise ValueError(
                    f"{self._benchmark_name} requires function_name for function-mode execution."
                )
            return self._score_function_mode(
                code=code,
                function_name=function_name,
                tests=tests,
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
                details=details,
            )

        return self._score_stdio_mode(
            code=code,
            language=language,
            tests=tests,
            checker=checker,
            checker_language=checker_language,
            checker_support_files=checker_support_files,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            details=details,
        )

    def _score_stdio_mode(
        self,
        *,
        code: str,
        language: str,
        tests: list[dict[str, str]],
        checker: str | None,
        checker_language: str,
        checker_support_files: dict[str, str],
        timeout_seconds: float | None,
        memory_limit_mb: float | None,
        details: dict[str, object],
    ) -> MetricScore:
        passed = 0
        failures: list[dict[str, object]] = []
        for index, test_case in enumerate(tests, start=1):
            execution = self._executor.execute(
                code=code,
                language=language,
                stdin=test_case["input"],
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            )
            if not execution.ok:
                failures.append(
                    {
                        "test_index": index,
                        "reason": "execution_failed",
                        "message": execution.message or execution.stderr,
                    }
                )
                continue
            if checker is not None:
                checker_files = {
                    _checker_filename(checker_language): checker,
                    **checker_support_files,
                    "input.txt": test_case["input"],
                    "correct_output.txt": test_case["output"],
                    "solution_output.txt": execution.stdout,
                }
                checker_result = self._executor.execute(
                    code=checker,
                    language=checker_language or "python",
                    files=checker_files,
                    args=["input.txt", "correct_output.txt", "solution_output.txt"],
                    timeout_seconds=timeout_seconds,
                    memory_limit_mb=memory_limit_mb,
                )
                if _checker_succeeded(
                    checker_result,
                    checker_language=checker_language or "python",
                ):
                    passed += 1
                    continue
                failures.append(
                    {
                        "test_index": index,
                        "reason": "checker_failed",
                        "message": checker_result.message or checker_result.stderr,
                    }
                )
                continue
            if _normalize_output(execution.stdout) == _normalize_output(
                test_case["output"]
            ):
                passed += 1
                continue
            failures.append(
                {
                    "test_index": index,
                    "reason": "wrong_answer",
                    "expected": test_case["output"],
                    "actual": execution.stdout,
                }
            )

        total_tests = len(tests)
        details["passed_tests"] = passed
        if failures:
            details["failures"] = failures[:3]
        return MetricScore(
            metric_id=self._metric_id,
            value=passed / total_tests if total_tests else 0.0,
            details=validate_json_dict(details, label="code execution score details"),
            error=None if passed == total_tests else "One or more tests failed.",
        )

    def _score_function_mode(
        self,
        *,
        code: str,
        function_name: str,
        tests: list[dict[str, str]],
        timeout_seconds: float | None,
        memory_limit_mb: float | None,
        details: dict[str, object],
    ) -> MetricScore:
        harness = _build_python_function_harness(code=code, function_name=function_name)
        passed = 0
        failures: list[dict[str, object]] = []
        for index, test_case in enumerate(tests, start=1):
            execution = self._executor.execute(
                code=harness,
                language="python",
                stdin=test_case["input"],
                timeout_seconds=timeout_seconds,
                memory_limit_mb=memory_limit_mb,
            )
            if not execution.ok:
                failures.append(
                    {
                        "test_index": index,
                        "reason": "execution_failed",
                        "message": execution.message or execution.stderr,
                    }
                )
                continue
            if _functional_output_matches(execution.stdout, test_case["output"]):
                passed += 1
                continue
            failures.append(
                {
                    "test_index": index,
                    "reason": "wrong_answer",
                    "expected": test_case["output"],
                    "actual": execution.stdout,
                }
            )

        total_tests = len(tests)
        details["passed_tests"] = passed
        if failures:
            details["failures"] = failures[:3]
        return MetricScore(
            metric_id=self._metric_id,
            value=passed / total_tests if total_tests else 0.0,
            details=validate_json_dict(details, label="code execution score details"),
            error=None if passed == total_tests else "One or more tests failed.",
        )

    def _resolve_checker_support_files(
        self,
        context: dict[str, object],
    ) -> dict[str, str]:
        resolved = dict(self._checker_support_files)
        custom_support_files = context.get("checker_support_files")
        if isinstance(custom_support_files, dict):
            for name, content in custom_support_files.items():
                if isinstance(name, str):
                    resolved[name] = str(content)
        return resolved


def _ordered_files(
    code: str,
    language: str,
    files: dict[str, str] | None,
) -> dict[str, str]:
    resolved = {str(name): str(content) for name, content in (files or {}).items()}
    main_name = resolved_main_filename(language)
    if main_name in resolved:
        return resolved
    return {main_name: code, **resolved}


def resolved_main_filename(language: str) -> str:
    normalized = language.strip().lower()
    if normalized == "python":
        return "main.py"
    return "main.cpp"


def _resolve_piston_runtime(language: str) -> tuple[str, str]:
    normalized = language.strip().lower()
    if normalized == "python":
        return (
            os.getenv("THEMIS_CODE_PISTON_PYTHON_LANGUAGE", "python"),
            os.getenv("THEMIS_CODE_PISTON_PYTHON_VERSION", "*"),
        )
    if normalized in {"cpp", "cplusplus"}:
        return (
            os.getenv("THEMIS_CODE_PISTON_CPP_LANGUAGE", "c++"),
            os.getenv("THEMIS_CODE_PISTON_CPP_VERSION", "*"),
        )
    raise ValueError(f"Unsupported Codeforces language '{language}'.")


def _resolve_sandbox_fusion_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized == "python":
        return "python"
    if normalized in {"cpp", "cplusplus"}:
        return "cpp"
    raise ValueError(f"Unsupported Codeforces language '{language}'.")


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
            payload = {"input": raw_input, "output": raw_output}
            raw_testtype = entry.get("testtype")
            if isinstance(raw_testtype, str) and raw_testtype.strip():
                payload["testtype"] = raw_testtype.strip().lower()
            tests.append(payload)
    return tests


def _normalize_checker(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_output(value: str) -> str:
    return value.replace("\r\n", "\n").strip()


def _checker_passed(stdout: str) -> bool:
    candidate = stdout.strip()
    if not candidate:
        return False
    try:
        return float(candidate) > 0.0
    except ValueError:
        return candidate.lower() in {"ok", "pass", "passed", "true"}


def _checker_succeeded(
    result: SandboxExecutionResult,
    *,
    checker_language: str,
) -> bool:
    if not result.ok:
        return False
    normalized_language = checker_language.strip().lower()
    if normalized_language in {"cpp", "cplusplus", "c++"}:
        return True
    if not result.stdout.strip():
        return True
    return _checker_passed(result.stdout)


def _checker_filename(language: str) -> str:
    normalized = language.strip().lower()
    if normalized == "python":
        return "checker.py"
    return "checker.cpp"


def _functional_output_matches(actual: str, expected: str) -> bool:
    actual_value = _parse_python_literal_or_text(_normalize_output(actual))
    expected_value = _parse_python_literal_or_text(_normalize_output(expected))
    return actual_value == expected_value


def _parse_python_literal_or_text(value: str) -> object:
    if not value:
        return ""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _build_python_function_harness(*, code: str, function_name: str) -> str:
    return (
        "import ast\n"
        "import sys\n"
        "from typing import *\n\n"
        f"{code}\n\n"
        "raw_input = sys.stdin.read().strip('\\n')\n"
        "arg_lines = [] if raw_input == '' else raw_input.split('\\n')\n"
        "args = []\n"
        "for line in arg_lines:\n"
        "    try:\n"
        "        args.append(ast.literal_eval(line))\n"
        "    except Exception:\n"
        "        args.append(line)\n"
        f"func_name = {function_name!r}\n"
        "target = None\n"
        "solution_cls = globals().get('Solution')\n"
        "if isinstance(solution_cls, type) and hasattr(solution_cls, func_name):\n"
        "    target = getattr(solution_cls(), func_name)\n"
        "elif func_name in globals() and callable(globals()[func_name]):\n"
        "    target = globals()[func_name]\n"
        "if target is None:\n"
        "    raise RuntimeError(f'Missing callable {func_name!r} in candidate code.')\n"
        "result = target(*args)\n"
        "print(repr(result))\n"
    )


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _milliseconds(seconds: float | None, *, default_ms: int) -> int:
    if seconds is None or seconds <= 0:
        return default_ms
    return max(1, int(seconds * 1000))


def _memory_bytes(memory_limit_mb: float | None) -> int:
    if memory_limit_mb is None or memory_limit_mb <= 0:
        return -1
    return int(memory_limit_mb * 1024 * 1024)


def _coalesce_message(*payloads: dict[str, object]) -> str | None:
    for payload in payloads:
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return None


def _default_executor() -> SandboxExecutor:
    backend = os.getenv("THEMIS_CODE_SANDBOX", "piston").strip().lower()
    if backend == "piston":
        return PistonSandboxExecutor()
    if backend == "sandbox_fusion":
        return SandboxFusionExecutor()
    raise ValueError("THEMIS_CODE_SANDBOX must be either 'piston' or 'sandbox_fusion'.")


def _sandbox_fusion_code(
    *,
    code: str,
    language: str,
    files: dict[str, str] | None,
    args: list[str] | None,
) -> str:
    normalized = language.strip().lower()
    resolved_args = list(args or [])
    if normalized == "python" and resolved_args and files and "checker.py" in files:
        argv_json = json.dumps(["checker.py", *resolved_args])
        return (
            "import sys\n"
            f"sys.argv = {argv_json}\n"
            "globals_dict = {'__name__': '__main__', '__file__': 'checker.py'}\n"
            "exec(compile(open('checker.py').read(), 'checker.py', 'exec'), globals_dict)\n"
        )
    return code


__all__ = [
    "CodeExecutionMetric",
    "PistonSandboxExecutor",
    "SandboxExecutionResult",
    "SandboxExecutor",
    "SandboxFusionExecutor",
    "_default_executor",
    "_resolve_piston_runtime",
    "resolved_main_filename",
]
