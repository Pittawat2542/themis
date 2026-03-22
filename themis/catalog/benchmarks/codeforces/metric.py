"""Execution metric for Open-R1 Codeforces."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib import request

from themis.records import MetricScore


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
            base_url
            or os.getenv("THEMIS_CODEFORCES_PISTON_URL")
            or "http://localhost:2000"
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
        with self._urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
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
            or os.getenv("THEMIS_CODEFORCES_SANDBOX_FUSION_URL")
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
        with self._urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
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


class CodeforcesExecutionMetric:
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        self._executor = executor or _default_executor()

    def score(self, trial, candidate, context):
        del trial
        code = ""
        inference = getattr(candidate, "inference", None)
        if inference is not None and getattr(inference, "raw_text", None) is not None:
            code = str(inference.raw_text)
        language = str(context.get("language", "")).strip().lower()
        input_mode = str(context.get("input_mode", "")).strip().lower()
        tests = _normalize_tests(context.get("official_tests"))
        checker = _normalize_checker(context.get("generated_checker"))
        timeout_seconds = _coerce_float(context.get("time_limit"))
        memory_limit_mb = _coerce_float(context.get("memory_limit"))
        details: dict[str, object] = {
            "language": language,
            "input_mode": input_mode,
            "total_tests": len(tests),
            "passed_tests": 0,
            "used_checker": bool(checker),
        }
        if input_mode != "stdio":
            raise ValueError(
                "codeforces only supports stdio rows during execution; "
                f"got input_mode='{input_mode or '<missing>'}'."
            )
        if language not in {"python", "cpp", "cplusplus"}:
            raise ValueError(
                "codeforces requires a supported language "
                f"('python', 'cpp', or 'cplusplus'); got "
                f"'{language or '<missing>'}'."
            )
        if not code.strip():
            details["error"] = "empty_candidate"
            return MetricScore(
                metric_id="codeforces_pass_rate",
                value=0.0,
                details=details,
                error="Candidate output did not contain runnable code.",
            )
        if not tests:
            raise ValueError(
                "codeforces requires official_tests for execution scoring."
            )

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
                checker_result = self._executor.execute(
                    code=checker,
                    language="python",
                    files={
                        "checker.py": checker,
                        "input.txt": test_case["input"],
                        "correct_output.txt": test_case["output"],
                        "solution_output.txt": execution.stdout,
                    },
                    args=["input.txt", "correct_output.txt", "solution_output.txt"],
                    timeout_seconds=timeout_seconds,
                    memory_limit_mb=memory_limit_mb,
                )
                if checker_result.ok and _checker_passed(checker_result.stdout):
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
            metric_id="codeforces_pass_rate",
            value=passed / total_tests if total_tests else 0.0,
            details=details,
            error=None if passed == total_tests else "One or more tests failed.",
        )


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
            os.getenv("THEMIS_CODEFORCES_PISTON_PYTHON_LANGUAGE", "python"),
            os.getenv("THEMIS_CODEFORCES_PISTON_PYTHON_VERSION", "*"),
        )
    if normalized in {"cpp", "cplusplus"}:
        return (
            os.getenv("THEMIS_CODEFORCES_PISTON_CPP_LANGUAGE", "c++"),
            os.getenv("THEMIS_CODEFORCES_PISTON_CPP_VERSION", "*"),
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
            tests.append({"input": raw_input, "output": raw_output})
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
    backend = os.getenv("THEMIS_CODEFORCES_SANDBOX", "piston").strip().lower()
    if backend == "piston":
        return PistonSandboxExecutor()
    if backend == "sandbox_fusion":
        return SandboxFusionExecutor()
    raise ValueError(
        "THEMIS_CODEFORCES_SANDBOX must be either 'piston' or 'sandbox_fusion'."
    )


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
