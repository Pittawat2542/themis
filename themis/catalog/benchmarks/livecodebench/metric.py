"""Execution metric for LiveCodeBench code generation."""

from __future__ import annotations

from ...runtime.code_execution import CodeExecutionMetric, SandboxExecutor


class LiveCodeBenchExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            metric_id="livecodebench_pass_rate",
            benchmark_name="livecodebench",
            supported_languages={"python"},
            supported_modes={"stdio", "function"},
            executor=executor,
        )
