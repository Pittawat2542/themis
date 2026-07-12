"""Code execution backends owned by Themis core."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field

from themis.core.base import FrozenModel


class CodeExecutionStatus(StrEnum):
    """Terminal status for a code execution attempt."""

    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class CodeExecutionLimits(FrozenModel):
    """Best-effort local execution limits."""

    timeout_seconds: float = 5.0
    max_output_chars: int = 20_000


class CodeExecutionRequest(FrozenModel):
    """Request to execute a single code artifact."""

    code: str
    language: str = "python"
    stdin: str = ""
    files: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    limits: CodeExecutionLimits = Field(default_factory=CodeExecutionLimits)


class CodeExecutionResult(FrozenModel):
    """Observed result from a code execution backend."""

    status: CodeExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0.0
    timed_out: bool = False
    backend_id: str
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is CodeExecutionStatus.OK and self.exit_code == 0


class UnsafeLocalSubprocessExecutor:
    """Run trusted Python with full host permissions in a subprocess."""

    backend_id = "unsafe_local_subprocess"

    def __init__(self, *, allow_unsafe: bool = False) -> None:
        if not allow_unsafe:
            raise ValueError(
                "UnsafeLocalSubprocessExecutor has full host permissions; "
                "pass allow_unsafe=True only for trusted code."
            )

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        language = request.language.strip().lower()
        if language not in {"python", "py"}:
            return CodeExecutionResult(
                status=CodeExecutionStatus.ERROR,
                backend_id=self.backend_id,
                message=f"Unsupported code language '{request.language}'.",
            )

        started = time.perf_counter()
        with TemporaryDirectory(prefix="themis-code-") as tmp:
            root = Path(tmp)
            try:
                self._write_files(root, request.files)
            except ValueError as exc:
                return CodeExecutionResult(
                    status=CodeExecutionStatus.ERROR,
                    backend_id=self.backend_id,
                    message=str(exc),
                    duration_seconds=time.perf_counter() - started,
                )
            main_path = root / "main.py"
            main_path.write_text(request.code, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(main_path), *request.args],
                    input=request.stdin,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    timeout=request.limits.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return CodeExecutionResult(
                    status=CodeExecutionStatus.TIMEOUT,
                    stdout=_trim_output(_decode_timeout_stream(exc.stdout), request),
                    stderr=_trim_output(_decode_timeout_stream(exc.stderr), request),
                    exit_code=None,
                    duration_seconds=time.perf_counter() - started,
                    timed_out=True,
                    backend_id=self.backend_id,
                    message="Execution timed out.",
                )

        status = (
            CodeExecutionStatus.OK
            if completed.returncode == 0
            else CodeExecutionStatus.FAILED
        )
        return CodeExecutionResult(
            status=status,
            stdout=_trim_output(completed.stdout, request),
            stderr=_trim_output(completed.stderr, request),
            exit_code=completed.returncode,
            duration_seconds=time.perf_counter() - started,
            backend_id=self.backend_id,
        )

    def _write_files(self, root: Path, files: dict[str, str]) -> None:
        for name, content in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe execution file path '{name}'.")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


class DockerExecutionBackend:
    """Optional Docker-backed executor for externally marked tests and workflows."""

    backend_id = "docker"

    def __init__(self, *, image: str = "python:3.12-slim") -> None:
        self._image = image

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if request.language.strip().lower() not in {"python", "py"}:
            return CodeExecutionResult(
                status=CodeExecutionStatus.ERROR,
                backend_id=self.backend_id,
                message=f"Unsupported code language '{request.language}'.",
            )
        if shutil.which("docker") is None:
            return CodeExecutionResult(
                status=CodeExecutionStatus.ERROR,
                backend_id=self.backend_id,
                message="Docker executable is not available.",
            )

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "--network",
                    "none",
                    "--read-only",
                    "--pids-limit",
                    "64",
                    "--memory",
                    "512m",
                    "--cpus",
                    "1.0",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                    "--user",
                    "65534:65534",
                    self._image,
                    "python",
                    "-c",
                    request.code,
                    *request.args,
                ],
                input=request.stdin,
                capture_output=True,
                text=True,
                timeout=request.limits.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeExecutionResult(
                status=CodeExecutionStatus.TIMEOUT,
                stdout=_trim_output(_decode_timeout_stream(exc.stdout), request),
                stderr=_trim_output(_decode_timeout_stream(exc.stderr), request),
                duration_seconds=time.perf_counter() - started,
                timed_out=True,
                backend_id=self.backend_id,
                message="Execution timed out.",
            )

        return CodeExecutionResult(
            status=(
                CodeExecutionStatus.OK
                if completed.returncode == 0
                else CodeExecutionStatus.FAILED
            ),
            stdout=_trim_output(completed.stdout, request),
            stderr=_trim_output(completed.stderr, request),
            exit_code=completed.returncode,
            duration_seconds=time.perf_counter() - started,
            backend_id=self.backend_id,
        )


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _trim_output(value: str, request: CodeExecutionRequest) -> str:
    limit = request.limits.max_output_chars
    if len(value) <= limit:
        return value
    return value[:limit]
