"""Execution metric for AetherCode."""

from __future__ import annotations

from pathlib import Path

from ..codeforces.metric import CodeExecutionMetric, SandboxExecutor


def _default_checker_support_files() -> dict[str, str]:
    testlib_path = Path(__file__).with_name("testlib.h")
    if not testlib_path.exists():
        return {}
    return {"testlib.h": testlib_path.read_text()}


class AetherCodeExecutionMetric(CodeExecutionMetric):
    def __init__(self, executor: SandboxExecutor | None = None) -> None:
        super().__init__(
            metric_id="aethercode_pass_rate",
            benchmark_name="aethercode",
            supported_languages={"cpp", "cplusplus"},
            supported_modes={"stdio"},
            executor=executor,
            checker_support_files=_default_checker_support_files(),
        )
