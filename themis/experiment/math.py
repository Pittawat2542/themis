"""High-level helpers for math-focused experiments."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, math_verify_utils, metrics, pipeline
from themis.experiment import orchestrator, storage as experiment_storage
from themis.generation import clients, plan, runner, templates


def build_math500_zero_shot_experiment(
    *,
    model_client: clients.FakeMathModelClient | None = None,
    model_name: str = "fake-math-llm",
    provider_name: str = "fake",
    temperature: float | None = None,
    sampling: core_entities.SamplingConfig | None = None,
    storage: experiment_storage.ExperimentStorage | None = None,
    runner_options: dict[str, object] | None = None,
) -> orchestrator.ExperimentOrchestrator:
    """Create an experiment orchestrator tailored for the MATH-500 benchmark."""

    prompt_template = templates.PromptTemplate(
        name="math-zero-shot-json",
        template=dedent(
            """
            You are an expert competition mathematician. Solve the following problem in a zero-shot
            manner. Think carefully and provide a short reasoning paragraph followed by a line of the
            form `Final Answer: \\boxed{{value}}` where `value` is the final numeric result.

            Problem:
            {problem}
            """
        ).strip(),
        metadata={"task": "math500", "expect_boxed": True},
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

    math_runner = runner.GenerationRunner(
        provider=model_client or clients.FakeMathModelClient(),
        **(runner_options or {}),
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
    exact = report.evaluation_report.metrics.get("ExactMatch")
    mean = exact.mean if exact else 0.0
    count = exact.count if exact else 0
    failures = len(report.failures) + len(report.evaluation_report.failures)
    return (
        f"Evaluated {report.metadata['total_samples']} samples | "
        f"exact match: {mean:.3f} over {count} generations | failures: {failures}"
    )


__all__ = [
    "build_math500_zero_shot_experiment",
    "run_math500_zero_shot",
    "summarize_report",
]
