"""Execution metric for Open-R1 Codeforces."""

from __future__ import annotations

from ...runtime.code_execution import (
    CodeExecutionMetric,
    PistonSandboxExecutor,
    SandboxExecutionResult,
    SandboxExecutor,
    SandboxFusionExecutor,
    _default_executor,
    _resolve_piston_runtime,
    resolved_main_filename,
)


class CodeforcesExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            metric_id="codeforces_pass_rate",
            benchmark_name="codeforces",
            supported_languages={"python", "cpp", "cplusplus"},
            supported_modes={"stdio"},
            executor=executor,
        )


__all__ = [
    "CodeExecutionMetric",
    "CodeforcesExecutionMetric",
    "PistonSandboxExecutor",
    "SandboxExecutionResult",
    "SandboxExecutor",
    "SandboxFusionExecutor",
    "_default_executor",
    "_resolve_piston_runtime",
    "resolved_main_filename",
]
