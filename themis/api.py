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

import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from themis.backends.execution import ExecutionBackend
    from themis.backends.storage import StorageBackend
    from themis.experiment.cache_manager import CacheManager

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
from themis.interfaces import DatasetAdapter
from themis.presets import parse_model_name
from themis.providers import create_provider
from themis.providers.options import normalize_provider_options
from themis.evaluation.metric_resolver import (
    get_registered_metrics,
    register_metric,
    resolve_metrics,
)

logger = logging.getLogger(__name__)


def _ensure_providers_registered() -> None:
    """Import provider modules to ensure they register themselves (lazy)."""
    try:
        from themis.generation import clients  # noqa: F401
        from themis.generation.providers import (
            litellm_provider,  # noqa: F401
            vllm_provider,  # noqa: F401
        )
    except ImportError:
        # We don't raise here as this is triggered on evaluate() call which might
        # not need these specific providers, but we log it if needed.
        pass


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
    "seed",
)
_ALLOWED_EXTRA_OPTIONS = {"top_p", *_PROVIDER_OPTION_KEYS}


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
    from themis.experiment.manifest import build_reproducibility_manifest
    from themis.experiment.cache_manager import CacheManager
    from themis.experiment.orchestrator import ExperimentOrchestrator

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

    # Import presets system (lazy import to avoid circular dependencies)
    from themis.presets import get_benchmark_preset

    # Determine if we're using a benchmark or custom dataset
    is_benchmark = isinstance(benchmark_or_dataset, str)

    if is_benchmark:
        if reference_field is not None:
            raise ConfigurationError(
                "`reference_field` is only supported for custom datasets."
            )
        benchmark_name = benchmark_or_dataset
        logger.info(f"Loading benchmark: {benchmark_name}")

        # Get preset configuration
        try:
            preset = get_benchmark_preset(benchmark_or_dataset)  # type: ignore
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
            prompt_template = preset.prompt_template  # type: ignore
        else:
            prompt_template = PromptTemplate(name="custom", template=prompt)

        # Use preset metrics if not overridden
        if metrics is None:
            metrics_list = preset.metrics
        else:
            metrics_list = resolve_metrics(metrics)

        # Use preset extractor
        extractor = preset.extractor

        # Use preset metadata fields
        metadata_fields = ()
        selected_reference_field = preset.reference_field
        dataset_id_field = preset.dataset_id_field
    else:
        # Custom dataset
        logger.info("Using custom dataset")
        dataset = list(benchmark_or_dataset)  # type: ignore
        logger.info(f"Custom dataset has {len(dataset)} samples")

        # Limit dataset if requested
        if limit is not None:
            dataset = dataset[:limit]
            logger.info(f"Limited to {len(dataset)} samples")

        # Use provided prompt or default
        if prompt is None:
            raise ConfigurationError(
                "Custom datasets require a prompt template. "
                "Example: prompt='Solve: {question}'"
            )
        prompt_template = PromptTemplate(name="custom", template=prompt)

        # Use provided metrics or defaults
        if metrics is None:
            metrics_list = resolve_metrics(["exact_match"])
        else:
            metrics_list = resolve_metrics(metrics)

        # Use identity extractor by default
        from themis.evaluation.extractors import IdentityExtractor

        extractor = IdentityExtractor()

        # Use standard field names
        metadata_fields = ()
        selected_reference_field = _resolve_custom_reference_field(
            dataset, requested_field=reference_field
        )
        if (
            _metrics_require_references(metrics_list)
            and selected_reference_field is None
        ):
            raise ConfigurationError(
                "Could not detect a reference field for custom dataset. "
                "Provide rows with a consistent `answer` or `reference` column, "
                "or pass `reference_field=...`."
            )
        dataset_id_field = "id"

    # Build evaluation pipeline
    pipeline = EvaluationPipeline(
        extractor=extractor,
        metrics=metrics_list,
    )
    logger.info(f"Evaluation metrics: {[m.name for m in metrics_list]}")

    # ========================================================================
    # Inline session logic (previously in ExperimentSession.run())
    # ========================================================================

    if not isinstance(pipeline, EvaluationPipelineContract):
        raise ConfigurationError("pipeline must implement EvaluationPipelineContract.")

    # Resolve storage
    resolved_storage_backend = _resolve_storage(
        storage, storage_backend=storage_backend
    )
    cache_manager = CacheManager(
        storage=resolved_storage_backend,
        enable_resume=resume,
        enable_cache=resume,
    )

    # Resolve dataset with cache support
    dataset_list = _resolve_dataset_with_cache(
        dataset,
        cache_manager=cache_manager,
        run_id=run_id,
        resume=resume,
    )

    # Parse model and build configuration
    provider_name, model_id, resolved_provider_options = _parse_model(
        model, provider_options=provider_options
    )
    model_spec = ModelSpec(identifier=model_id, provider=provider_name)
    sampling = _build_sampling(
        temperature=temperature,
        top_p=kwargs.get("top_p", 0.95),
        max_tokens=max_tokens,
    )

    # Build generation plan
    plan = GenerationPlan(
        templates=[PromptTemplate(name="default", template=prompt_template.template)],
        models=[model_spec],
        sampling_parameters=[sampling],
        dataset_id_field=dataset_id_field,
        reference_field=selected_reference_field,
        metadata_fields=metadata_fields,
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
    manifest = build_reproducibility_manifest(
        model=model_id,
        provider=provider_name,
        provider_options=resolved_provider_options,
        sampling={
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": max_tokens,
        },
        num_samples=num_samples,
        evaluation_config=_build_evaluation_config(pipeline),
        seeds={
            "provider_seed": resolved_provider_options.get("seed"),
            "sampling_seed": None,
        },
        dataset_fingerprint=_dataset_fingerprint(dataset_list),
        prompt_fingerprint=_prompt_fingerprint(prompt_template.template),
    )

    # Create orchestrator and run
    orchestrator = ExperimentOrchestrator(
        generation_plan=plan,
        generation_runner=runner,
        evaluation_pipeline=pipeline,
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


def _extract_provider_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    options = {key: kwargs[key] for key in _PROVIDER_OPTION_KEYS if key in kwargs}
    return normalize_provider_options(options)


def _detect_reference_field(dataset: Sequence[dict[str, Any]]) -> str | None:
    if not dataset:
        return "answer"
    answer_in_all = all("answer" in row for row in dataset)
    reference_in_all = all("reference" in row for row in dataset)
    if answer_in_all:
        return "answer"
    if reference_in_all:
        return "reference"
    if any(("answer" in row) or ("reference" in row) for row in dataset):
        raise ConfigurationError(
            "Detected mixed or partial reference fields across rows. "
            "Use a consistent `answer` or `reference` column, or pass "
            "`reference_field=...`."
        )
    return None


def _resolve_custom_reference_field(
    dataset: Sequence[dict[str, Any]], *, requested_field: str | None
) -> str | None:
    if requested_field is None:
        return _detect_reference_field(dataset)
    if requested_field == "":
        return None
    missing_indices = [
        str(idx + 1) for idx, row in enumerate(dataset) if requested_field not in row
    ]
    if missing_indices:
        preview = ", ".join(missing_indices[:5])
        raise ConfigurationError(
            f"reference_field='{requested_field}' is missing in row(s): {preview}."
        )
    return requested_field


def _metrics_require_references(resolved_metrics: Sequence[Any]) -> bool:
    return any(
        getattr(metric, "requires_reference", True) for metric in resolved_metrics
    )


def _should_warn_missing_api_key(model: str, provider_options: dict[str, Any]) -> bool:
    if provider_options.get("api_key"):
        return False
    provider = _detect_provider_name(model).lower()
    if provider in {"fake", "vllm"}:
        return False
    if provider_options.get("api_base"):
        # Assume custom/local endpoints can be keyless.
        return False
    key_envs = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "LITELLM_API_KEY",
    )
    if any(os.getenv(env) for env in key_envs):
        return False
    return provider in {
        "litellm",
        "openai",
        "anthropic",
        "azure",
        "bedrock",
        "gemini",
        "cohere",
    }


def _detect_provider_name(model: str) -> str:
    if ":" in model:
        return model.split(":", 1)[0]
    from themis.presets import parse_model_name

    provider_name, _, _ = parse_model_name(model)
    return provider_name


# ============================================================================
# Helper functions inlined from session.py
# ============================================================================


def _parse_model(
    model: str, *, provider_options: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Parse model string into provider, model_id, and options."""
    options = normalize_provider_options(provider_options)
    if ":" in model:
        provider_name, model_id = model.split(":", 1)
        return provider_name, model_id, options

    parsed_provider, model_id, parsed_options = parse_model_name(model, **options)
    return parsed_provider, model_id, normalize_provider_options(parsed_options)


def _build_sampling(
    temperature: float, top_p: float, max_tokens: int
) -> SamplingConfig:
    """Build SamplingConfig from individual parameters."""
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _resolve_dataset_with_cache(
    dataset: DatasetAdapter | Iterable | Sequence[dict] | object,
    *,
    cache_manager: "CacheManager",
    run_id: str | None,
    resume: bool,
) -> list[dict]:
    """Resolve dataset, checking cache if needed."""
    # 1. Try resolving direct dataset
    resolved = []
    if isinstance(dataset, DatasetAdapter):
        resolved = list(dataset.iter_samples())
    elif isinstance(dataset, Iterable):
        resolved = list(dataset)  # type: ignore[arg-type]

    if resolved:
        return resolved

    # 2. Try loading from cache if resume is enabled
    if resume and run_id:
        cached = cache_manager.load_cached_dataset(run_id)
        if cached:
            return cached

    # 3. Fail if nothing found
    raise ConfigurationError(
        "No dataset provided. Supply `dataset=` to evaluate() "
        "or ensure `run_id` points to a cached run."
    )


def _resolve_storage(storage_path: str | Path | None, storage_backend: Any | None):
    """Resolve storage backend from path or explicit backend."""
    if storage_backend is not None:
        backend = storage_backend
        if hasattr(backend, "experiment_storage"):
            return backend.experiment_storage
        if not hasattr(backend, "start_run"):
            raise ConfigurationError(
                "storage_backend must be ExperimentStorage-compatible."
            )
        return backend
    root = (
        Path(storage_path) if storage_path is not None else Path(".cache/experiments")
    )
    from themis.storage import ExperimentStorage

    return ExperimentStorage(root)


def _build_evaluation_config(pipeline: EvaluationPipelineContract) -> dict[str, Any]:
    """Build evaluation config fingerprint from pipeline."""
    if hasattr(pipeline, "evaluation_fingerprint"):
        try:
            fingerprint = dict(pipeline.evaluation_fingerprint())
        except Exception:
            fingerprint = {}
    else:
        fingerprint = {}

    if "metrics" not in fingerprint and hasattr(pipeline, "_metrics"):
        fingerprint["metrics"] = sorted(
            [
                f"{metric.__class__.__module__}.{metric.__class__.__name__}:{metric.name}"
                for metric in pipeline._metrics
            ]
        )
    if "extractor" not in fingerprint and hasattr(pipeline, "_extractor"):
        extractor = pipeline._extractor
        fingerprint["extractor"] = (
            f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"
        )
        if "extractor_field" not in fingerprint and hasattr(extractor, "field_name"):
            fingerprint["extractor_field"] = extractor.field_name
    fingerprint.setdefault("metrics", [])
    fingerprint.setdefault("extractor", "unknown")

    return fingerprint


def _stable_json_hash(value: object) -> str:
    """Create stable hash of JSON-serializable value."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_fingerprint(dataset: Sequence[dict]) -> str:
    """Create fingerprint for dataset."""
    return _stable_json_hash(dataset)


def _prompt_fingerprint(prompt: str) -> str:
    """Create fingerprint for prompt template."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


__all__ = ["evaluate", "register_metric", "get_registered_metrics"]
