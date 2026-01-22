"""Unified API for Themis - The primary interface for all evaluations.

This module provides the main entry point for running evaluations:
    - Simple one-liner for benchmarks
    - Custom datasets with minimal configuration
    - Distributed execution and cloud storage support
    - Auto-configuration of prompts, metrics, and extractors

Example:
    ```python
    import themis
    
    # Simple benchmark evaluation
    report = themis.evaluate("math500", model="gpt-4", limit=100)
    
    # Custom dataset
    report = themis.evaluate(
        dataset=[{"id": "1", "question": "...", "answer": "..."}],
        model="claude-3-opus",
        prompt="Solve: {question}"
    )
    
    # Distributed with cloud storage
    report = themis.evaluate(
        "gsm8k",
        model="gpt-4",
        distributed=True,
        workers=8,
        storage="s3://my-bucket/experiments"
    )
    ```
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from themis.core.entities import (
    ExperimentReport,
    GenerationRecord,
    ModelSpec,
    PromptSpec,
    SamplingConfig,
)
from themis.evaluation.pipeline import EvaluationPipeline
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.generation.plan import GenerationPlan
from themis.generation.router import ProviderRouter
from themis.generation.runner import GenerationRunner
from themis.generation.templates import PromptTemplate
from themis.providers import create_provider


def evaluate(
    benchmark_or_dataset: str | Sequence[dict[str, Any]],
    *,
    model: str,
    limit: int | None = None,
    prompt: str | None = None,
    metrics: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_samples: int = 1,
    distributed: bool = False,
    workers: int = 4,
    storage: str | Path | None = None,
    run_id: str | None = None,
    resume: bool = True,
    on_result: Callable[[GenerationRecord], None] | None = None,
    **kwargs: Any,
) -> ExperimentReport:
    """Run an LLM evaluation with automatic configuration.
    
    This is the primary API for Themis. It auto-configures prompts, metrics,
    and extractors based on the benchmark name, or allows full customization
    for custom datasets.
    
    Args:
        benchmark_or_dataset: Either a benchmark name (e.g., "math500", "gsm8k")
            or a list of dataset samples as dictionaries. For custom datasets,
            each dict should have: prompt/question (input), answer/reference (output),
            and optionally id (unique identifier).
        model: Model identifier for LiteLLM (e.g., "gpt-4", "claude-3-opus-20240229",
            "azure/gpt-4", "ollama/llama3"). Provider is auto-detected from the name.
        limit: Maximum number of samples to evaluate. Use for testing or when you
            want to evaluate a subset. None means evaluate all samples.
        prompt: Custom prompt template using Python format strings. Variables like
            {prompt}, {question}, {context} will be replaced with dataset fields.
            If None, uses the benchmark's default prompt template.
        metrics: List of metric names to compute. Available: "ExactMatch", "MathVerify",
            "BLEU", "ROUGE", "BERTScore", "METEOR", "PassAtK", "CodeBLEU",
            "ExecutionAccuracy". If None, uses benchmark defaults.
        temperature: Sampling temperature (0.0 = deterministic/greedy, 1.0 = standard,
            2.0 = very random). Recommended: 0.0 for evaluation reproducibility.
        max_tokens: Maximum tokens in model response. Typical values: 256 for short
            answers, 512 for medium, 2048 for long explanations or code.
        num_samples: Number of responses to generate per prompt. Use >1 for Pass@K
            metrics, ensembling, or measuring response variance.
        distributed: Whether to use distributed execution. Currently a placeholder
            for future Ray integration.
        workers: Number of parallel workers for generation. Higher = faster but may
            hit rate limits. Recommended: 4-16 for APIs, 32+ for local models.
        storage: Storage location for results and cache. Defaults to ".cache/experiments".
            Can be a local path or (future) cloud storage URI.
        run_id: Unique identifier for this run. If None, auto-generated from timestamp
            (e.g., "run-2024-01-15-123456"). Use meaningful IDs for tracking experiments.
        resume: Whether to resume from cached results.
        on_result: Optional callback function called for each result.
        **kwargs: Additional provider-specific options.
    
    Returns:
        ExperimentReport containing generation results, evaluation metrics,
        and metadata.
    
    Raises:
        ValueError: If benchmark is unknown or configuration is invalid.
        RuntimeError: If evaluation fails.
    
    Example:
        >>> report = themis.evaluate("math500", model="gpt-4", limit=10)
        >>> print(f"Accuracy: {report.evaluation_report.metrics['accuracy']:.2%}")
        Accuracy: 85.00%
    """
    # Import presets system (lazy import to avoid circular dependencies)
    from themis.presets import get_benchmark_preset, parse_model_name
    
    # Determine if we're using a benchmark or custom dataset
    is_benchmark = isinstance(benchmark_or_dataset, str)
    
    if is_benchmark:
        benchmark_name = benchmark_or_dataset
        # Get preset configuration
        preset = get_benchmark_preset(benchmark_name)
        
        # Load dataset using preset loader
        dataset = preset.load_dataset(limit=limit)
        
        # Use preset prompt if not overridden
        if prompt is None:
            prompt_template = preset.prompt_template
        else:
            prompt_template = PromptTemplate(name="custom", template=prompt)
        
        # Use preset metrics if not overridden
        if metrics is None:
            metrics_list = preset.metrics
        else:
            metrics_list = _resolve_metrics(metrics)
        
        # Use preset extractor
        extractor = preset.extractor
        
        # Use preset metadata fields
        metadata_fields = preset.metadata_fields
        reference_field = preset.reference_field
        dataset_id_field = preset.dataset_id_field
    else:
        # Custom dataset
        dataset = list(benchmark_or_dataset)
        
        # Limit dataset if requested
        if limit is not None:
            dataset = dataset[:limit]
        
        # Use provided prompt or default
        if prompt is None:
            raise ValueError(
                "Custom datasets require a prompt template. "
                "Example: prompt='Solve: {question}'"
            )
        prompt_template = PromptTemplate(name="custom", template=prompt)
        
        # Use provided metrics or defaults
        if metrics is None:
            metrics_list = _resolve_metrics(["exact_match"])
        else:
            metrics_list = _resolve_metrics(metrics)
        
        # Use identity extractor by default
        from themis.evaluation.extractors import IdentityExtractor
        extractor = IdentityExtractor()
        
        # Use standard field names
        metadata_fields = ()
        reference_field = "answer"
        dataset_id_field = "id"
    
    # Parse model name to get provider and options
    provider_name, model_id, provider_options = parse_model_name(model, **kwargs)
    
    # Create model spec
    model_spec = ModelSpec(
        identifier=model_id,
        provider=provider_name,
    )
    
    # Create sampling config
    sampling_config = SamplingConfig(
        temperature=temperature,
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=max_tokens,
    )
    
    # Create generation plan
    plan = GenerationPlan(
        templates=[prompt_template],
        models=[model_spec],
        sampling_parameters=[sampling_config],
        dataset_id_field=dataset_id_field,
        reference_field=reference_field,
        metadata_fields=metadata_fields,
    )
    
    # Create provider and router
    provider = create_provider(provider_name, **provider_options)
    router = ProviderRouter({model_id: provider})
    
    # Create runner
    runner = GenerationRunner(provider=router)
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        extractor=extractor,
        metrics=metrics_list,
    )
    
    # Determine storage location
    if storage is None:
        storage_dir = Path.home() / ".themis" / "runs"
    else:
        storage_dir = Path(storage) if not str(storage).startswith(("s3://", "gs://", "azure://")) else storage
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create storage backend
    if isinstance(storage_dir, Path):
        from themis.experiment.storage import ExperimentStorage
        storage_backend = ExperimentStorage(storage_dir)
    else:
        # Cloud storage (to be implemented in Phase 3)
        raise NotImplementedError(
            f"Cloud storage not yet implemented. Use local path for now. "
            f"Requested: {storage_dir}"
        )
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner,
        evaluation_pipeline=pipeline,
        storage=storage_backend,
    )
    
    # Run evaluation
    if distributed:
        # Distributed execution (to be implemented in Phase 3)
        raise NotImplementedError(
            "Distributed execution not yet implemented. "
            "Set distributed=False to use local execution."
        )
    
    # Run locally
    report = orchestrator.run(
        dataset=dataset,
        max_samples=limit,
        run_id=run_id,
        resume=resume,
        on_result=on_result,
    )
    
    return report


def _resolve_metrics(metric_names: list[str]) -> list:
    """Resolve metric names to metric instances.
    
    Args:
        metric_names: List of metric names (e.g., ["exact_match", "bleu"])
    
    Returns:
        List of metric instances
    
    Raises:
        ValueError: If a metric name is unknown
    """
    from themis.evaluation.metrics.exact_match import ExactMatch
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy
    from themis.evaluation.metrics.response_length import ResponseLength
    
    # NLP metrics (Phase 2)
    try:
        from themis.evaluation.metrics.nlp import BLEU, ROUGE, BERTScore, METEOR, ROUGEVariant
        nlp_available = True
    except ImportError:
        nlp_available = False
    
    # Metric registry
    METRICS_REGISTRY = {
        # Core metrics
        "exact_match": ExactMatch,
        "math_verify": MathVerifyAccuracy,
        "response_length": ResponseLength,
    }
    
    # Add NLP metrics if available
    if nlp_available:
        METRICS_REGISTRY.update({
            "bleu": BLEU,
            "rouge1": lambda: ROUGE(variant=ROUGEVariant.ROUGE_1),
            "rouge2": lambda: ROUGE(variant=ROUGEVariant.ROUGE_2),
            "rougeL": lambda: ROUGE(variant=ROUGEVariant.ROUGE_L),
            "bertscore": BERTScore,
            "meteor": METEOR,
        })
    
    # Code metrics (to be added later in Phase 2)
    # "pass_at_k": PassAtK,
    # "codebleu": CodeBLEU,
    
    metrics = []
    for name in metric_names:
        if name not in METRICS_REGISTRY:
            available = ", ".join(sorted(METRICS_REGISTRY.keys()))
            raise ValueError(
                f"Unknown metric: {name}. "
                f"Available metrics: {available}"
            )
        
        metric_cls = METRICS_REGISTRY[name]
        # Handle both class and lambda factory
        if callable(metric_cls) and not isinstance(metric_cls, type):
            metrics.append(metric_cls())
        else:
            metrics.append(metric_cls())
    
    return metrics


__all__ = ["evaluate"]
