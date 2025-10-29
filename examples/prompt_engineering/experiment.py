"""Implementation for running the prompt engineering experiment."""

from __future__ import annotations

from themis.core import entities as core_entities
from themis.evaluation import extractors, metrics
from themis.experiment import builder as experiment_builder
from themis.experiment import orchestrator
from themis.generation import templates
from themis.project import ProjectExperiment
from themis.utils.progress import ProgressReporter

# Import provider modules to ensure they're registered
from themis.generation.providers import litellm_provider  # noqa: F401
from themis.generation.providers import vllm_provider  # noqa: F401

from . import datasets as dataset_loader
from .config import PromptEngineeringConfig, ModelConfig
from .prompts import create_prompt_templates


def create_project_experiment(config: PromptEngineeringConfig) -> ProjectExperiment:
    """Create a project experiment from the prompt engineering configuration."""

    # Create prompt templates from config
    prompt_templates = create_prompt_templates(config)

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=prompt_templates,
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",  # Field in dataset that contains expected answer
        metadata_fields=("dataset_name", "subject", "level", "prompt_strategy"),
        context_builder=lambda row: {
            "problem": row["problem"], 
            "question": row.get("question", row["problem"])  # Support both field names
        },
    )

    # Create project experiment
    project_experiment = ProjectExperiment(
        name="prompt-engineering-experiment",
        description="Systematic evaluation of different prompting strategies",
        definition=definition,
        metadata={
            "storage_dir": config.storage_dir,
            "run_id": config.run_id,
            "resume": config.resume,
        },
    )

    return project_experiment


def run_experiment(config: PromptEngineeringConfig) -> orchestrator.ExperimentReport:
    """Run the prompt engineering experiment."""
    
    # Load dataset rows
    dataset_rows: list[dict[str, object]] = []
    for dataset_cfg in config.datasets:
        rows = dataset_loader.load_dataset(dataset_cfg)
        dataset_rows.extend(rows)

    if not dataset_rows:
        raise ValueError("Prompt engineering experiment requires at least one dataset row")

    # Create prompt templates from config
    prompt_templates = create_prompt_templates(config)

    sampling_parameters = [profile.to_sampling_config() for profile in config.samplings]
    model_bindings = [_make_binding(model_cfg) for model_cfg in config.models]

    definition = experiment_builder.ExperimentDefinition(
        templates=prompt_templates,
        sampling_parameters=sampling_parameters,
        model_bindings=model_bindings,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("dataset_name", "subject", "level", "prompt_strategy"),
        context_builder=lambda row: {
            "problem": row["problem"], 
            "question": row.get("question", row["problem"])
        },
    )

    # Set up experiment builder with appropriate extractor and metrics
    # For math problems, we'll use a more flexible extractor since local models might not always return JSON
    builder = experiment_builder.ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),  # Try JSON first
        # Or use a simple text extractor as fallback in case JSON isn't returned
        metrics=[
            metrics.ExactMatch(case_sensitive=False, strip_whitespace=True),
            metrics.ResponseLength(),
        ],
    )

    built = builder.build(definition, storage_dir=config.storage_dir)

    # Calculate total tasks for progress reporting
    total_tasks = 0
    for row in dataset_rows:
        # Each row will be processed for each combination of:
        # - prompt template
        # - model
        # - sampling configuration
        total_tasks += len(config.prompt_variants) * len(config.models) * len(config.samplings)

    print(f"Running experiment with {total_tasks} total tasks...")
    print(f"  - {len(config.prompt_variants)} prompt variants")
    print(f"  - {len(config.models)} models")
    print(f"  - {len(config.samplings)} sampling strategies")
    print(f"  - {len(dataset_rows)} dataset samples")

    with ProgressReporter(total=total_tasks, description="Generating") as progress:
        report = built.orchestrator.run(
            dataset=dataset_rows,
            run_id=config.run_id,
            resume=config.resume,
            on_result=progress.on_result,
        )
    return report


def summarize_report(report: orchestrator.ExperimentReport) -> str:
    """Create a summary of the prompt engineering experiment results."""
    
    # Extract metric information
    exact_match_metric = report.evaluation_report.metrics.get("ExactMatch")
    response_length_metric = report.evaluation_report.metrics.get("ResponseLength")
    
    exact_mean = exact_match_metric.mean if exact_match_metric else 0.0
    exact_count = exact_match_metric.count if exact_match_metric else 0
    
    response_length_mean = response_length_metric.mean if response_length_metric else 0.0

    # Get failure counts
    generation_failures = len(report.failures)
    evaluation_failures = len(report.evaluation_report.failures)
    total_failures = generation_failures + evaluation_failures

    # Get metadata
    total_samples = report.metadata.get("total_samples", 0)
    successful_generations = report.metadata.get("successful_generations", 0)
    failed_generations = report.metadata.get("failed_generations", 0)
    
    # Calculate total tasks (samples × prompt variants × models × sampling strategies)
    # We need to extract this information from the report or config
    # For now, we'll calculate based on what we know about the structure
    total_tasks = len(report.generation_results)

    # Build summary string
    summary_parts = [
        f"Evaluated {total_samples} samples across all prompt variations and models",
        f"Successful generations: {successful_generations}/{total_tasks}",
        f"Exact match accuracy: {exact_mean:.3f} ({exact_count} evaluated)",
    ]
    
    if response_length_mean > 0:
        summary_parts.append(f"Average response length: {response_length_mean:.1f} characters")

    # Add failure information
    if total_failures > 0:
        summary_parts.append(
            f"Failures: {total_failures} (gen: {failed_generations}, eval: {evaluation_failures})"
        )
    else:
        summary_parts.append("No failures")

    return " | ".join(summary_parts)


def analyze_by_prompt_strategy(report: orchestrator.ExperimentReport):
    """Analyze results by prompt strategy to compare effectiveness."""
    
    # Group results by prompt strategy
    strategy_results = {}
    
    for record in report.generation_results:
        # Extract prompt strategy from template metadata
        prompt_name = record.task.prompt.template_name
        strategy = record.task.prompt.metadata.get("strategy", "unknown")
        
        if strategy not in strategy_results:
            strategy_results[strategy] = {
                "count": 0,
                "correct": 0,
                "metrics": {}
            }
        
        strategy_results[strategy]["count"] += 1
        
        # Check if this record has exact match score
        exact_match_score = record.metrics.get("ExactMatch")
        if exact_match_score and exact_match_score.value == 1.0:
            strategy_results[strategy]["correct"] += 1
    
    # Print comparison
    print("\nPrompt Strategy Comparison:")
    print("-" * 50)
    for strategy, data in strategy_results.items():
        accuracy = data["correct"] / data["count"] if data["count"] > 0 else 0
        print(f"{strategy:20} | Accuracy: {accuracy:.3f} ({data['correct']}/{data['count']})")
    
    return strategy_results


def _make_binding(model_cfg: ModelConfig) -> experiment_builder.ModelBinding:
    """Create a model binding from configuration."""
    spec = core_entities.ModelSpec(
        identifier=model_cfg.name,
        provider=model_cfg.provider,
        metadata={"description": model_cfg.description or model_cfg.name},
    )
    return experiment_builder.ModelBinding(
        spec=spec,
        provider_name=model_cfg.provider,
        provider_options=model_cfg.provider_options,
    )


__all__ = [
    "run_experiment", 
    "summarize_report", 
    "create_project_experiment",
    "analyze_by_prompt_strategy"
]