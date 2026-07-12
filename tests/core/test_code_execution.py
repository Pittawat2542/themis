from __future__ import annotations

from themis.core.code_execution import (
    CodeExecutionLimits,
    CodeExecutionRequest,
    CodeExecutionStatus,
    UnsafeLocalSubprocessExecutor,
)
import pytest


def test_unsafe_local_executor_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_unsafe=True"):
        UnsafeLocalSubprocessExecutor()


def test_local_subprocess_backend_executes_python_successfully() -> None:
    backend = UnsafeLocalSubprocessExecutor(allow_unsafe=True)

    result = backend.execute(
        CodeExecutionRequest(
            code="import sys\nprint('hello ' + sys.stdin.read().strip())",
            language="python",
            stdin="themis",
        )
    )

    assert result.status is CodeExecutionStatus.OK
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello themis"
    assert result.stderr == ""
    assert result.timed_out is False


def test_local_subprocess_backend_reports_failed_processes() -> None:
    backend = UnsafeLocalSubprocessExecutor(allow_unsafe=True)

    result = backend.execute(
        CodeExecutionRequest(
            code="import sys\nprint('bad', file=sys.stderr)\nsys.exit(3)",
            language="python",
        )
    )

    assert result.status is CodeExecutionStatus.FAILED
    assert result.exit_code == 3
    assert result.stderr.strip() == "bad"


def test_local_subprocess_backend_reports_timeouts() -> None:
    backend = UnsafeLocalSubprocessExecutor(allow_unsafe=True)

    result = backend.execute(
        CodeExecutionRequest(
            code="import time\ntime.sleep(1)",
            language="python",
            limits=CodeExecutionLimits(timeout_seconds=0.05),
        )
    )

    assert result.status is CodeExecutionStatus.TIMEOUT
    assert result.timed_out is True
    assert result.exit_code is None


def test_local_subprocess_backend_supports_files_and_args() -> None:
    backend = UnsafeLocalSubprocessExecutor(allow_unsafe=True)

    result = backend.execute(
        CodeExecutionRequest(
            code=(
                "from pathlib import Path\n"
                "import sys\n"
                "print(Path('fixture.txt').read_text().strip() + ':' + sys.argv[1])"
            ),
            language="python",
            files={"fixture.txt": "value\n"},
            args=["arg"],
        )
    )

    assert result.status is CodeExecutionStatus.OK
    assert result.stdout.strip() == "value:arg"


def test_local_subprocess_backend_rejects_unsupported_languages() -> None:
    backend = UnsafeLocalSubprocessExecutor(allow_unsafe=True)

    result = backend.execute(
        CodeExecutionRequest(code="int main() { return 0; }", language="cpp")
    )

    assert result.status is CodeExecutionStatus.ERROR
    assert "Unsupported code language" in (result.message or "")
