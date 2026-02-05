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

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Sequence

from themis.core.entities import ExperimentReport, GenerationRecord
from themis.evaluation.pipeline import EvaluationPipeline
from themis.generation.templates import PromptTemplate
from themis.session import ExperimentSession
from themis.specs import ExperimentSpec, ExecutionSpec, StorageSpec

# Import provider modules to ensure they register themselves
try:
    from themis.generation import clients  # noqa: F401 - registers fake provider
    from themis.generation.providers import (
        litellm_provider,  # noqa: F401
        vllm_provider,  # noqa: F401
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


# Module-level metrics registry for custom metrics
_METRICS_REGISTRY: dict[str, type] = {}
_PROVIDER_OPTION_KEYS = (
    "api_key",
    "base_url",
    "api_base",
    "api_version",
    "timeout",
    "max_retries",
    "n_parallel",
    "organization",
    "api_type",
    "region_name",
)


def register_metric(name: str, metric_cls: type) -> None:
    """Register a custom metric for use in evaluate().
    
    This allows users to add their own metrics to Themis without modifying
    the source code. Registered metrics can be used by passing their names
    to the `metrics` parameter in evaluate().
    
    Args:
        name: Metric name (used in evaluate(metrics=[name]))
        metric_cls: Metric class implementing the Metric interface.
            Must have a compute() method that takes prediction, references,
            and metadata parameters.
    
    Raises:
        TypeError: If metric_cls is not a class
        ValueError: If metric_cls doesn't implement the required interface
    
    Example:
        >>> from themis.evaluation.metrics import MyCustomMetric
        >>> themis.register_metric("my_metric", MyCustomMetric)
        >>> report = themis.evaluate("math500", model="gpt-4", metrics=["my_metric"])
    """
    if not isinstance(metric_cls, type):
        raise TypeError(f"metric_cls must be a class, got {type(metric_cls)}")
    
    # Validate that it implements the Metric interface
    if not hasattr(metric_cls, "compute"):
        raise ValueError(
            f"{metric_cls.__name__} must implement compute() method. "
            f"See themis.evaluation.metrics for examples."
        )
    
    _METRICS_REGISTRY[name] = metric_cls
    logger.info(f"Registered custom metric: {name} -> {metric_cls.__name__}")


def get_registered_metrics() -> dict[str, type]:
    """Get all currently registered custom metrics.
    
    Returns:
        Dictionary mapping metric names to their classes
    """
    return _METRICS_REGISTRY.copy()


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
    max_records_in_memory: int | None = None,
    distributed: bool = False,
    workers: int = 4,
    storage: str | Path | None = None,
    storage_backend: object | None = None,
    execution_backend: object | None = None,
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
        max_records_in_memory: Optional cap on generation/evaluation records kept in
            the returned report to bound memory for very large runs.
        distributed: Whether to use distributed execution. Currently a placeholder
            for future Ray integration.
        workers: Number of parallel workers for generation. Higher = faster but may
            hit rate limits. Recommended: 4-16 for APIs, 32+ for local models.
        storage: Storage location for results and cache. Defaults to ".cache/experiments".
            Can be a local path or (future) cloud storage URI.
        storage_backend: Optional storage backend instance. Typically an
            ExperimentStorage or LocalFileStorageBackend (adapter). Custom
            storage backends are not yet integrated with the evaluate() API.
        execution_backend: Optional execution backend for custom parallelism.
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
    if distributed:
        raise ValueError(
            "distributed=True is not supported yet. "
            "Use execution_backend for custom/distributed execution."
        )

    logger.info("=" * 60)
    logger.info("Starting Themis evaluation")
    logger.info(f"Model: {model}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    if num_samples > 1:
        logger.info(f"Num samples per prompt: {num_samples}")
    if "api_base" in kwargs:
        logger.info(f"Custom API base: {kwargs['api_base']}")
    if "api_key" in kwargs:
        logger.info("API key: <provided>")
    else:
        logger.warning("⚠️  No api_key provided - may fail for custom API endpoints")
    logger.info("=" * 60)
    
    provider_options = _extract_provider_options(kwargs)

    # Import presets system (lazy import to avoid circular dependencies)
    from themis.presets import get_benchmark_preset
    
    # Determine if we're using a benchmark or custom dataset
    is_benchmark = isinstance(benchmark_or_dataset, str)
    
    if is_benchmark:
        benchmark_name = benchmark_or_dataset
        logger.info(f"Loading benchmark: {benchmark_name}")
        
        # Get preset configuration
        try:
            preset = get_benchmark_preset(benchmark_name)
        except Exception as e:
            logger.error(f"❌ Failed to get benchmark preset '{benchmark_name}': {e}")
            raise
        
        # Load dataset using preset loader
        logger.info(f"Loading dataset (limit={limit})...")
        try:
            dataset = preset.load_dataset(limit=limit)
            logger.info(f"✅ Loaded {len(dataset)} samples from {benchmark_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
        
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
        logger.info("Using custom dataset")
        dataset = list(benchmark_or_dataset)
        logger.info(f"Custom dataset has {len(dataset)} samples")
        
        # Limit dataset if requested
        if limit is not None:
            dataset = dataset[:limit]
            logger.info(f"Limited to {len(dataset)} samples")
        
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
        reference_field = _detect_reference_field(dataset)
        dataset_id_field = "id"
    
    # Build evaluation pipeline
    pipeline = EvaluationPipeline(
        extractor=extractor,
        metrics=metrics_list,
    )
    logger.info(f"Evaluation metrics: {[m.name for m in metrics_list]}")

    # Compose vNext spec
    spec = ExperimentSpec(
        dataset=dataset,
        prompt=prompt_template.template,
        model=model,
        sampling={
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": max_tokens,
        },
        provider_options=provider_options,
        num_samples=num_samples,
        max_records_in_memory=max_records_in_memory,
        dataset_id_field=dataset_id_field,
        reference_field=reference_field,
        metadata_fields=metadata_fields,
        pipeline=pipeline,
        run_id=run_id,
    )

    execution = ExecutionSpec(
        backend=execution_backend,
        workers=workers,
    )

    storage_spec = StorageSpec(
        backend=storage_backend,
        path=storage,
        cache=resume,
    )

    session = ExperimentSession()
    return session.run(spec, execution=execution, storage=storage_spec, on_result=on_result)


def _extract_provider_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    options = {key: kwargs[key] for key in _PROVIDER_OPTION_KEYS if key in kwargs}
    # Normalize common alias for LiteLLM constructor compatibility.
    if "base_url" in options and "api_base" not in options:
        options["api_base"] = options.pop("base_url")
    elif "base_url" in options:
        options.pop("base_url")
    return options


def _detect_reference_field(dataset: Sequence[dict[str, Any]]) -> str | None:
    if not dataset:
        return "answer"
    if any("answer" in row for row in dataset):
        return "answer"
    if any("reference" in row for row in dataset):
        return "reference"
    return None


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

    # Code metrics (some optional dependencies)
    try:
        from themis.evaluation.metrics.code.execution import ExecutionAccuracy
        from themis.evaluation.metrics.code.pass_at_k import PassAtK
        code_metrics: dict[str, Any] = {
            "pass_at_k": PassAtK,
            "execution_accuracy": ExecutionAccuracy,
        }
        try:
            from themis.evaluation.metrics.code.codebleu import CodeBLEU
            code_metrics["codebleu"] = CodeBLEU
        except ImportError:
            pass
    except ImportError:
        code_metrics = {}
    
    # Built-in metrics registry
    BUILTIN_METRICS = {
        # Core metrics
        "exact_match": ExactMatch,
        "math_verify": MathVerifyAccuracy,
        "response_length": ResponseLength,
    }
    
    # Add NLP metrics if available
    if nlp_available:
        BUILTIN_METRICS.update({
            "bleu": BLEU,
            "rouge1": lambda: ROUGE(variant=ROUGEVariant.ROUGE_1),
            "rouge2": lambda: ROUGE(variant=ROUGEVariant.ROUGE_2),
            "rougeL": lambda: ROUGE(variant=ROUGEVariant.ROUGE_L),
            "bertscore": BERTScore,
            "meteor": METEOR,
        })

    BUILTIN_METRICS.update(code_metrics)
    
    # Merge built-in and custom metrics
    # Custom metrics can override built-in metrics
    METRICS_REGISTRY = {**BUILTIN_METRICS, **_METRICS_REGISTRY}
    
    def _normalize_metric_name(name: str) -> str | None:
        raw = name.strip()
        if raw in METRICS_REGISTRY:
            return raw
        lowered = raw.lower()
        if lowered in METRICS_REGISTRY:
            return lowered
        for key in METRICS_REGISTRY.keys():
            if key.lower() == lowered:
                return key
        # Convert CamelCase / PascalCase to snake_case
        import re

        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
        if snake in METRICS_REGISTRY:
            return snake
        return None

    metrics = []
    for name in metric_names:
        resolved = _normalize_metric_name(name)
        if resolved is None:
            available = ", ".join(sorted(METRICS_REGISTRY.keys()))
            raise ValueError(
                f"Unknown metric: {name}. "
                f"Available metrics: {available}"
            )
        
        metric_cls = METRICS_REGISTRY[resolved]
        # Handle both class and lambda factory
        if callable(metric_cls) and not isinstance(metric_cls, type):
            metrics.append(metric_cls())
        else:
            metrics.append(metric_cls())
    
    return metrics


__all__ = ["evaluate", "register_metric", "get_registered_metrics"]
