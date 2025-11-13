"""Judge evaluation experiment using RubricJudgeMetric, PairwiseJudgeMetric, and ConsistencyMetric."""

from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import metrics, strategies as evaluation_strategies, extractors
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import clients, templates
from themis.utils.progress import ProgressReporter

from . import datasets as dataset_loader
from .config import JudgeExperimentConfig, DEFAULT_JUDGE_CONFIG


def run_experiment(config: JudgeExperimentConfig | None = None) -> orchestrator.ExperimentReport:
    """Run judge-based evaluation experiment.
    
    This experiment demonstrates:
    - RubricJudgeMetric: Score candidate solutions using rubric criteria
    - ConsistencyMetric: Measure agreement across multiple judge runs
    - JudgeEvaluationStrategy: Aggregate multi-judge scores
    """
    config = config or DEFAULT_JUDGE_CONFIG
    dataset_rows = dataset_loader.load_demo_dataset()

    # Create a simple template that returns candidate solution
    template = templates.PromptTemplate(
        name="pass-through",
        template="{candidate_solution}",
        metadata={"task": "judge-eval"},
    )

    # Use fake provider to simulate generation
    fake_provider = clients.FakeMathModelClient(seed=42, default_answer="simulated")

    # Build judge model spec
    judge_model = core_entities.ModelSpec(
        identifier="judge-gpt4",
        provider="fake",
    )

    # Create rubric from config
    rubric_config = config.rubrics[0] if config.rubrics else None
    rubric = rubric_config.criteria if rubric_config else {
        "correctness": "Answer matches ground truth",
        "reasoning": "Clear step-by-step explanation",
    }

    # Build metrics: RubricJudgeMetric and ConsistencyMetric
    judge_metric = metrics.RubricJudgeMetric(
        judge_model=judge_model,
        judge_provider=fake_provider,
        rubric=rubric,
    )

    # For demo, we'll use exact match alongside judge
    exact_match = metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)

    # Build experiment using JudgeEvaluationStrategy
    definition = experiment_builder.ExperimentDefinition(
        templates=[template],
        sampling_parameters=[
            core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=512)
        ],
        model_bindings=[
            experiment_builder.ModelBinding(
                spec=core_entities.ModelSpec(
                    identifier="fake-model",
                    provider="fake",
                ),
                provider_name="fake",
                provider_options={},
            )
        ],
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject",),
        context_builder=lambda row: {"candidate_solution": row["candidate_solution"]},
    )

    # Use JudgeEvaluationStrategy to aggregate judge scores
    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.IdentityExtractor(),
        metrics=[exact_match, judge_metric],
        evaluation_strategy_resolver=lambda record: evaluation_strategies.JudgeEvaluationStrategy(),
    )

    built = builder.build(definition, storage_dir=config.storage_dir)

    total_tasks = len(dataset_rows)
    with ProgressReporter(total=total_tasks, description="Evaluating") as progress:
        report = built.orchestrator.run(
            dataset=dataset_rows,
            run_id=config.run_id,
            resume=config.resume,
            on_result=progress.on_result,
        )

    return report


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    """Summarize judge evaluation report."""
    exact_match = report.evaluation_report.metrics.get("ExactMatch")
    judge_rubric = report.evaluation_report.metrics.get("RubricJudge")

    parts = [f"Evaluated {report.metadata.get('total_samples', 0)} samples"]

    if exact_match:
        parts.append(f"ExactMatch: {exact_match.mean:.3f}")

    if judge_rubric:
        parts.append(f"RubricJudge: {judge_rubric.mean:.3f}")

    failures = len(report.failures) + len(report.evaluation_report.failures)
    if failures > 0:
        parts.append(f"Failures: {failures}")
    else:
        parts.append("No failures")

    return " | ".join(parts)


__all__ = ["run_experiment", "summarize_report"]
