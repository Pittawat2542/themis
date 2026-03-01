"""Unified API for Themis - The primary interface for all evaluations.

This module provides the main entry point for running evaluations:
    - Built-in benchmarks (e.g., "math500", "gsm8k")
    - Custom datasets (list of dictionaries) with minimal configuration
    - Distributed execution and cloud storage support
    - Auto-configuration of prompts, metrics, and extractors

# Quick Examples

**1. Simple Benchmark Evaluation**
Run an existing benchmark (like 'math500') using a specific model.
Themis automatically configures the correct metrics and prompt templates.

```python
import themis

# Run 100 samples from MATH-500 using OpenAI's GPT-4o
report = themis.evaluate("math500", model="openai/gpt-4o", limit=100)

print(f"Accuracy: {report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
```

**2. Custom Dataset Evaluation**
Evaluate your own data by passing a list of dictionaries. You must supply a prompt template and choose your metrics.

```python
import themis

dataset = [
    {"id": "q1", "question": "What is 2+2?", "answer": "4"},
    {"id": "q2", "question": "What is the capital of France?", "answer": "Paris"}
]

# Provide a prompt string that references fields in your dataset dictionaries
report = themis.evaluate(
    dataset=dataset,
    model="anthropic/claude-3-5-sonnet-20241022",
    prompt="Answer the following question concisely: {question}",
    metrics=["exact_match"], # Default is exact_match if not provided
)
```

**3. Execution Options: Parallelism and Resilience**
Scale your evaluation natively using Python threads and automatic retries.

```python
report = themis.evaluate(
    "gsm8k",
    model="gpt-4o-mini",
    workers=16,          # Run 16 concurrent requests (watch your API limits!)
    max_retries=5,       # Retry up to 5 times on rate limits or API errors
    timeout=30,          # Set custom LiteLLM timeouts via kwargs
)
```
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from themis.backends.execution import ExecutionBackend
    from themis.backends.storage import StorageBackend

from themis.core.entities import (
    ExperimentReport,
    GenerationRecord,
    ModelSpec,
    SamplingConfig,
)
from themis.evaluation.pipeline import EvaluationPipeline, EvaluationPipelineContract
from themis.exceptions import ConfigurationError
from themis.generation.plan import GenerationPlan
from themis.generation.router import ProviderRouter
from themis.generation.runner import GenerationRunner
from themis.generation.strategies import RepeatedSamplingStrategy
from themis.generation.templates import PromptTemplate
from themis.providers import create_provider, parse_model
from themis.api._helpers import (
    _build_orchestrator,
    _build_run_manifest,
    _extract_provider_options,
    _resolve_dataset_with_cache,
    _resolve_evaluation_context,
    _should_warn_missing_api_key,
    _wire_storage,
    _ALLOWED_EXTRA_OPTIONS,
)
from themis.evaluation.metric_resolver import (
    get_registered_metrics,
    register_metric,
)

logger = logging.getLogger(__name__)


def _ensure_providers_registered() -> None:
    """Import provider modules to ensure they register themselves (lazy)."""
    try:
        from themis.generation import clients  # noqa: F401
        from themis.providers import (
            litellm_provider,  # noqa: F401
            vllm_provider,  # noqa: F401
        )
    except ImportError:
        # We don't raise here as this is triggered on evaluate() call which might
        # not need these specific providers, but we log it if needed.
        pass


def evaluate(
    benchmark_or_dataset: str | Sequence[dict[str, Any]],
    *,
    model: str,
    limit: int | None = None,
    prompt: str | None = None,
    reference_field: str | None = None,
    metrics: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_samples: int = 1,
    max_records_in_memory: int | None = None,
    workers: int = 4,
    max_retries: int = 3,
    storage: str | Path | None = None,
    storage_backend: StorageBackend | None = None,  # ExperimentStorage-compatible
    execution_backend: ExecutionBackend | None = None,  # ExecutionBackend-compatible
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
        benchmark_or_dataset: Either a built-in benchmark name (e.g., `"math500"`, `"gsm8k"`)
            or a list of dictionaries for custom datasets. For custom datasets,
            each dict should represent a single test case (e.g., `{"question": "...", "answer": "..."}`).
        model: Model identifier for LiteLLM (e.g., `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet-20241022"`,
            `"fake:fake-math-llm"`). Provider prefixes are strongly recommended.
        limit: Maximum number of samples to evaluate. Useful for quick testing before
            running a full benchmark. If `None`, evaluates all samples.
        prompt: Custom prompt template using Python format strings. Variables like
            `{question}` will be substituted with the corresponding keys from the dataset.
            Required if evaluating a custom dataset.
        reference_field: The dictionary key containing the expected answer (the "gold" reference).
            If None, Themis auto-detects common fields (`"answer"` or `"reference"`).
            Required if your custom dataset uses a different key (e.g., `reference_field="solution"`).
            This option is ignored for built-in benchmarks.
        metrics: List of metric names to compute. Common built-ins include:
            `"exact_match"`, `"math_verify"`, `"response_length"`, `"bleu"`, `"rougeL"`.
            If None, built-in benchmarks use their defaults; custom datasets default to `["exact_match"]`.
        temperature: Sampling temperature (0.0 = deterministic/greedy, 1.0 = standard).
            Recommended: 0.0 for reproducible evaluation loops.
        max_tokens: Maximum tokens in the model's generated response. Typical values:
            256 for short factual answers, 2048 for long reasoning paths (CoT) or code.
        num_samples: Number of responses to generate per prompt. Use >1 for Pass@K
            metrics or to measure generation consistency.
        max_records_in_memory: Optional cap on records kept in the returned report to
            bound memory for very large runs.
        workers: Number of parallel requests to send to the provider. Higher = faster, but
            increases the risk of HTTP 429 Rate Limit errors. Recommended: 8-16 for public APIs.
        max_retries: Number of automatic retries for generation failures (default: 3). Failed
            generations will ultimately result in empty text and `0.0` metric scores without crashing the run.
        storage: Local directory path to cache results. Defaults to `".cache/experiments"`.
        storage_backend: Optional advanced storage backend instance for custom databases.
        execution_backend: Optional advanced execution backend for distributed workers.
        run_id: Unique string identifier for caching this run. If you stop and restart
            a script with the same `run_id`, it will instantly resume from where it left off.
        resume: Whether to resume from cached results found in the storage directory under `run_id`.
        on_result: Optional callback function triggered immediately after each sample is evaluated.
        **kwargs: Additional provider-specific options passed directly to LiteLLM
            (e.g., `api_key`, `base_url`, `timeout`, `top_p`).

    Returns:
        ExperimentReport containing generation results, evaluation metrics,
        and metadata. You can inspect `.evaluation_report.metrics` for aggregated
        scores, or iterate over `.generation_records` to inspect individual answers.

    Raises:
        ConfigurationError: If benchmark is unknown or configuration is invalid.
        EvaluationError: If evaluation fails catastrophically. Note: individual item
            failures (e.g., occasional API errors after max_retries) will *not* throw
            an error, but will surface as empty outputs in the returned `ExperimentReport`.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> report = themis.evaluate("math500", model="openai/gpt-4o-mini", limit=10)
        >>> print(f"ExactMatch: {report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
        ExactMatch: 85.00%
    """
    _ensure_providers_registered()

    # Lazy imports to break circular: api → experiment → config → experiment
    logger.info("=" * 60)
    logger.info("Starting Themis evaluation")
    logger.info(f"Model: {model}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    if num_samples > 1:
        logger.info(f"Num samples per prompt: {num_samples}")
    if "api_base" in kwargs:
        logger.info(f"Custom API base: {kwargs['api_base']}")

    unsupported_options = sorted(set(kwargs.keys()) - _ALLOWED_EXTRA_OPTIONS)
    if unsupported_options:
        unsupported = ", ".join(unsupported_options)
        raise ConfigurationError(
            f"Unsupported option(s): {unsupported}. "
            "Supported extra options are provider options and `top_p`."
        )

    provider_options = _extract_provider_options(kwargs)
    if "api_key" in kwargs:
        logger.info("API key: <provided>")
    elif _should_warn_missing_api_key(model, provider_options):
        logger.warning("⚠️  No api_key provided - may fail for hosted API models")
    logger.info("=" * 60)

    # Resolve evaluation context
    eval_ctx = _resolve_evaluation_context(
        benchmark_or_dataset, limit, prompt, reference_field, metrics
    )

    # Build evaluation pipeline
    pipeline = EvaluationPipeline(
        extractor=eval_ctx.extractor,
        metrics=eval_ctx.metrics_list,
    )
    logger.info(f"Evaluation metrics: {[m.name for m in eval_ctx.metrics_list]}")

    # ========================================================================
    # Inline session logic (previously in ExperimentSession.run())
    # ========================================================================

    if not isinstance(pipeline, EvaluationPipelineContract):
        raise ConfigurationError("pipeline must implement EvaluationPipelineContract.")

    # Resolve storage
    cache_manager = _wire_storage(storage, storage_backend, resume)

    # Resolve dataset with cache support
    dataset_list = _resolve_dataset_with_cache(
        eval_ctx.dataset,
        cache_manager=cache_manager,
        run_id=run_id,
        resume=resume,
    )

    # Parse model and build configuration
    provider_name, model_id, resolved_provider_options = parse_model(
        model, **provider_options
    )
    model_spec = ModelSpec(identifier=model_id, provider=provider_name)
    sampling = SamplingConfig.from_params(
        temperature=temperature,
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=max_tokens,
    )

    # Build generation plan
    plan = GenerationPlan(
        templates=[
            PromptTemplate(name="default", template=eval_ctx.prompt_template.template)
        ],
        models=[model_spec],
        sampling_parameters=[sampling],
        dataset_id_field=eval_ctx.dataset_id_field,
        reference_field=eval_ctx.reference_field,
        metadata_fields=eval_ctx.metadata_fields,
    )

    # Create provider and router
    provider = create_provider(provider_name, **resolved_provider_options)
    router = ProviderRouter({(provider_name, model_id): provider})

    # Setup strategy resolver for multi-sampling
    strategy_resolver = (
        (lambda task: RepeatedSamplingStrategy(attempts=num_samples))  # noqa: ARG005
        if num_samples > 1
        else None
    )

    # Build generation runner
    runner = GenerationRunner(
        executor=router,
        strategy_resolver=strategy_resolver,
        max_parallel=workers,
        max_retries=max_retries,
        execution_backend=execution_backend,
    )

    # Build reproducibility manifest
    manifest = _build_run_manifest(
        model_id=model_id,
        provider_name=provider_name,
        provider_options=resolved_provider_options,
        temperature=temperature,
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=max_tokens,
        num_samples=num_samples,
        pipeline=pipeline,
        dataset_list=dataset_list,
        prompt_template=eval_ctx.prompt_template.template,
    )

    # Create orchestrator and run
    orchestrator = _build_orchestrator(
        plan=plan,
        runner=runner,
        pipeline=pipeline,
        cache_manager=cache_manager,
    )

    return orchestrator.run(
        dataset=dataset_list,
        run_id=run_id,
        resume=resume,
        cache_results=resume,
        on_result=on_result,
        run_manifest=manifest,
        max_records_in_memory=max_records_in_memory,
    )


__all__ = ["evaluate", "register_metric", "get_registered_metrics"]
