"""High-level helpers for math-focused experiments."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, math_verify_utils, metrics, pipeline
from themis.experiment import orchestrator
from themis.experiment import storage as experiment_storage
from themis.generation import clients, plan, runner, templates
from themis.interfaces import ModelProvider


def build_math500_zero_shot_experiment(
    *,
    model_client: ModelProvider | None = None,
    model_name: str = "fake-math-llm",
    provider_name: str = "fake",
    temperature: float | None = None,
    sampling: core_entities.SamplingConfig | None = None,
    storage: experiment_storage.ExperimentStorage | None = None,
    runner_options: dict[str, object] | None = None,
    task_name: str = "math500",
) -> orchestrator.ExperimentOrchestrator:
    """Create an experiment orchestrator tailored for competition math benchmarks."""

    prompt_template = templates.PromptTemplate(
        name=f"{task_name}-zero-shot-json",
        template=dedent(
            """
            You are an expert competition mathematician. Solve the following problem in a zero-shot
            manner. Think carefully and provide a short reasoning paragraph followed by a line of the
            form `Final Answer: \\boxed{{value}}` where `value` is the final numeric result.

            Problem:
            {problem}
            """
        ).strip(),
        metadata={"task": task_name, "expect_boxed": True},
    )

    sampling = sampling or core_entities.SamplingConfig(
        temperature=temperature if temperature is not None else 0.0,
        top_p=0.95,
        max_tokens=512,
    )
    model_spec = core_entities.ModelSpec(
        identifier=model_name, provider=provider_name, default_sampling=sampling
    )
    math_plan = plan.GenerationPlan(
        templates=[prompt_template],
        models=[model_spec],
        sampling_parameters=[sampling],
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject", "level"),
        context_builder=lambda row: {"problem": row.get("problem", "")},
    )

    # Extract runner options with proper type conversion
    runner_kwargs = {}
    if runner_options:
        # Convert values to appropriate types with type checking
        if (
            "max_parallel" in runner_options
            and runner_options["max_parallel"] is not None
        ):
            runner_kwargs["max_parallel"] = int(str(runner_options["max_parallel"]))
        if (
            "max_retries" in runner_options
            and runner_options["max_retries"] is not None
        ):
            runner_kwargs["max_retries"] = int(str(runner_options["max_retries"]))
        if (
            "retry_initial_delay" in runner_options
            and runner_options["retry_initial_delay"] is not None
        ):
            runner_kwargs["retry_initial_delay"] = float(
                str(runner_options["retry_initial_delay"])
            )
        if (
            "retry_backoff_multiplier" in runner_options
            and runner_options["retry_backoff_multiplier"] is not None
        ):
            runner_kwargs["retry_backoff_multiplier"] = float(
                str(runner_options["retry_backoff_multiplier"])
            )
        if "retry_max_delay" in runner_options:
            retry_max_delay = runner_options["retry_max_delay"]
            runner_kwargs["retry_max_delay"] = (
                float(str(retry_max_delay)) if retry_max_delay is not None else None
            )

    math_runner = runner.GenerationRunner(
        provider=model_client or clients.FakeMathModelClient(),
        **runner_kwargs,
    )
    if math_verify_utils.math_verify_available():
        extractor = extractors.MathVerifyExtractor()
        metric_list = [
            metrics.MathVerifyAccuracy(),
            metrics.ExactMatch(case_sensitive=False, strip_whitespace=True),
        ]
    else:
        extractor = extractors.JsonFieldExtractor(field_path="answer")
        metric_list = [
            metrics.ExactMatch(case_sensitive=False, strip_whitespace=True),
        ]
    eval_pipeline = pipeline.EvaluationPipeline(
        extractor=extractor,
        metrics=metric_list,
    )

    return orchestrator.ExperimentOrchestrator(
        generation_plan=math_plan,
        generation_runner=math_runner,
        evaluation_pipeline=eval_pipeline,
        storage=storage,
    )


def run_math500_zero_shot(
    dataset: Sequence[dict[str, object]],
    *,
    model_client: clients.FakeMathModelClient | None = None,
    max_samples: int | None = None,
    storage: experiment_storage.ExperimentStorage | None = None,
    run_id: str | None = None,
    resume: bool = True,
) -> orchestrator.ExperimentReport:
    """Run the zero-shot math experiment against a prepared dataset."""

    experiment = build_math500_zero_shot_experiment(
        model_client=model_client, storage=storage
    )
    return experiment.run(
        dataset, max_samples=max_samples, run_id=run_id, resume=resume
    )


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    # Get exact match metric
    exact = report.evaluation_report.metrics.get("ExactMatch")
    exact_mean = exact.mean if exact else 0.0
    exact_count = exact.count if exact else 0

    # Get MathVerify metric if available
    math_verify = report.evaluation_report.metrics.get("MathVerifyAccuracy")
    math_verify_mean = math_verify.mean if math_verify else None
    math_verify_count = math_verify.count if math_verify else 0

    # Get failure counts
    generation_failures = len(report.failures)
    evaluation_failures = len(report.evaluation_report.failures)
    total_failures = generation_failures + evaluation_failures

    # Get metadata
    total_samples = report.metadata.get("total_samples", 0)
    successful_generations = report.metadata.get("successful_generations", 0)
    failed_generations = report.metadata.get("failed_generations", 0)

    # Build summary string
    summary_parts = [
        f"Evaluated {total_samples} samples",
        f"Successful generations: {successful_generations}/{total_samples}",
        f"Exact match: {exact_mean:.3f} ({exact_count} evaluated)",
    ]

    # Add MathVerify accuracy if available
    if math_verify_mean is not None:
        summary_parts.append(
            f"MathVerify accuracy: {math_verify_mean:.3f} ({math_verify_count} evaluated)"
        )

    # Add failure information
    if total_failures > 0:
        summary_parts.append(
            f"Failures: {total_failures} (gen: {failed_generations}, eval: {evaluation_failures})"
        )
    else:
        summary_parts.append("No failures")

    return " | ".join(summary_parts)


__all__ = [
    "build_math500_zero_shot_experiment",
    "run_math500_zero_shot",
    "summarize_report",
]
