"""Catalog metric registration helpers."""

from __future__ import annotations

from functools import partial

from themis import BenchmarkDefinitionConfig, PluginRegistry


def register_mcq(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    if not registry.has_metric("choice_accuracy"):
        from ..runtime import ChoiceAccuracyMetric

        registry.register_metric("choice_accuracy", ChoiceAccuracyMetric)


def register_math(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    if not registry.has_metric("math_equivalence"):
        from ..runtime import MathEquivalenceMetric

        registry.register_metric("math_equivalence", MathEquivalenceMetric)


def register_simpleqa(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from ..benchmarks.simpleqa_verified.metric import SimpleQAVerifiedJudgeMetric

    _register_judge_metric(
        registry,
        metric_id="simpleqa_verified_score",
        metric_factory=partial(
            SimpleQAVerifiedJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_healthbench(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from ..benchmarks.healthbench.metric import HealthBenchRubricMetric

    _register_judge_metric(
        registry,
        metric_id="healthbench_score",
        metric_factory=partial(
            HealthBenchRubricMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_lpfqa(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from ..benchmarks.lpfqa.metric import LPFQAJudgeMetric

    _register_judge_metric(
        registry,
        metric_id="lpfqa_score",
        metric_factory=partial(
            LPFQAJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_frontierscience(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from ..benchmarks.frontierscience.metric import FrontierScienceJudgeMetric

    _register_judge_metric(
        registry,
        metric_id="frontierscience_score",
        metric_factory=partial(
            FrontierScienceJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_hle(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from ..benchmarks.hle.metric import HLEJudgeMetric

    _register_judge_metric(
        registry,
        metric_id="hle_accuracy",
        metric_factory=partial(
            HLEJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_codeforces(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.codeforces.metric import CodeforcesExecutionMetric

    _register_code_execution_metric(
        registry,
        metric_id="codeforces_pass_rate",
        metric_factory=CodeforcesExecutionMetric,
    )


def register_aethercode(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.aethercode.metric import AetherCodeExecutionMetric

    _register_code_execution_metric(
        registry,
        metric_id="aethercode_pass_rate",
        metric_factory=AetherCodeExecutionMetric,
    )


def register_livecodebench(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.livecodebench.metric import LiveCodeBenchExecutionMetric

    _register_code_execution_metric(
        registry,
        metric_id="livecodebench_pass_rate",
        metric_factory=LiveCodeBenchExecutionMetric,
    )


def register_humaneval(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.humaneval.metric import HumanEvalExecutionMetric

    if not registry.has_metric("humaneval_pass_rate"):
        registry.register_metric(
            "humaneval_pass_rate",
            partial(HumanEvalExecutionMetric, metric_id="humaneval_pass_rate"),
        )


def register_humaneval_plus(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.humaneval.metric import HumanEvalExecutionMetric

    if not registry.has_metric("humaneval_plus_pass_rate"):
        registry.register_metric(
            "humaneval_plus_pass_rate",
            partial(HumanEvalExecutionMetric, metric_id="humaneval_plus_pass_rate"),
        )


def register_procbench(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    from ..benchmarks.procbench.metric import ProcbenchFinalAccuracyMetric

    if not registry.has_metric("procbench_final_accuracy"):
        registry.register_metric(
            "procbench_final_accuracy",
            ProcbenchFinalAccuracyMetric,
        )


def _register_code_execution_metric(
    registry: PluginRegistry,
    *,
    metric_id: str,
    metric_factory: type,
) -> None:
    if not registry.has_metric(metric_id):
        registry.register_metric(metric_id, metric_factory)


def _register_judge_metric(
    registry: PluginRegistry,
    *,
    metric_id: str,
    metric_factory,
) -> None:
    if not registry.has_metric(metric_id):
        registry.register_metric(metric_id, metric_factory)
