"""Experiment builders for multiple-choice benchmarks."""

from __future__ import annotations

from textwrap import dedent
from typing import Callable, Sequence

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics, pipeline
from themis.experiment import orchestrator
from themis.experiment import storage as experiment_storage
from themis.generation import clients, plan, runner, templates
from themis.interfaces import ModelProvider


def build_multiple_choice_json_experiment(
    *,
    dataset_name: str,
    task_id: str | None = None,
    model_client: ModelProvider | None = None,
    model_name: str = "fake-math-llm",
    provider_name: str = "fake",
    temperature: float | None = None,
    sampling: core_entities.SamplingConfig | None = None,
    storage: experiment_storage.ExperimentStorage | None = None,
    runner_options: dict[str, object] | None = None,
    metadata_fields: Sequence[str] = ("subject",),
    context_builder: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> orchestrator.ExperimentOrchestrator:
    """Create an experiment orchestrator for multiple-choice QA benchmarks."""

    task_id = task_id or dataset_name
    prompt_template = templates.PromptTemplate(
        name=f"{dataset_name}-multiple-choice-json",
        template=dedent(
            """
            You are an expert test taker. Select the single best answer to the following
            multiple-choice question.

            Question:
            {question}

            Choices:
            {choices_block}

            Respond with a JSON object containing two keys:
              "answer" - the capital letter of the chosen option (e.g. "A")
              "explanation" - one or two sentences explaining your reasoning

            Example response:
            {{"answer": "A", "explanation": "Reasoning..."}}
            """
        ).strip(),
        metadata={"task": task_id, "response_format": "json"},
    )

    sampling = sampling or core_entities.SamplingConfig(
        temperature=temperature if temperature is not None else 0.0,
        top_p=0.95,
        max_tokens=512,
    )
    model_spec = core_entities.ModelSpec(
        identifier=model_name, provider=provider_name, default_sampling=sampling
    )

    def _default_context_builder(row: dict[str, object]) -> dict[str, object]:
        labels: Sequence[str] = tuple(
            str(label) for label in row.get("choice_labels", [])
        ) or tuple("ABCD")
        choices: Sequence[str] = tuple(str(choice) for choice in row.get("choices", []))
        choice_lines = []
        for label, choice in zip(labels, choices, strict=False):
            choice_lines.append(f"{label}. {choice}")
        choices_block = "\n".join(choice_lines)
        return {
            "question": str(row.get("question", "")),
            "choices_block": choices_block,
        }

    mcq_plan = plan.GenerationPlan(
        templates=[prompt_template],
        models=[model_spec],
        sampling_parameters=[sampling],
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=tuple(metadata_fields),
        context_builder=context_builder or _default_context_builder,
    )

    runner_kwargs = {}
    if runner_options:
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

    mcq_runner = runner.GenerationRunner(
        provider=model_client or clients.FakeMathModelClient(),
        **runner_kwargs,
    )

    extractor = extractors.JsonFieldExtractor(field_path="answer")
    metric_list = [
        metrics.ExactMatch(case_sensitive=False, strip_whitespace=True),
    ]
    eval_pipeline = pipeline.EvaluationPipeline(
        extractor=extractor,
        metrics=metric_list,
    )

    return orchestrator.ExperimentOrchestrator(
        generation_plan=mcq_plan,
        generation_runner=mcq_runner,
        evaluation_pipeline=eval_pipeline,
        storage=storage,
    )


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    exact = report.evaluation_report.metrics.get("ExactMatch")
    accuracy = exact.mean if exact else 0.0
    evaluated = exact.count if exact else 0

    total_samples = report.metadata.get("total_samples", evaluated)
    successful_generations = report.metadata.get("successful_generations", evaluated)
    failed_generations = report.metadata.get("failed_generations", 0)
    evaluation_failures = len(report.evaluation_report.failures)
    total_failures = failed_generations + evaluation_failures

    summary_parts = [
        f"Evaluated {total_samples} samples",
        f"Successful generations: {successful_generations}/{total_samples}",
        f"Accuracy: {accuracy:.3f} ({evaluated} evaluated)",
    ]
    if total_failures:
        summary_parts.append(
            f"Failures: {total_failures} (gen: {failed_generations}, eval: {evaluation_failures})"
        )
    else:
        summary_parts.append("No failures")
    return " | ".join(summary_parts)


__all__ = ["build_multiple_choice_json_experiment", "summarize_report"]
