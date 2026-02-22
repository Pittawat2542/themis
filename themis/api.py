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

    # Custom execution backend
    report = themis.evaluate(
        "gsm8k",
        model="gpt-4",
        workers=8,
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
from typing import Any, Iterable

from themis.core.entities import (
    ExperimentReport,
    GenerationRecord,
    ModelSpec,
    SamplingConfig,
)
from themis.evaluation.pipeline import EvaluationPipeline, EvaluationPipelineContract
from themis.experiment.manifest import build_reproducibility_manifest
from themis.experiment.cache_manager import CacheManager
from themis.experiment.orchestrator import ExperimentOrchestrator
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
        reference_field: Field name containing references for custom datasets.
            If None, Themis auto-detects a consistent field (`answer` or `reference`).
            This option is ignored for built-in benchmarks.
        metrics: List of metric names to compute. Common built-ins include:
            "exact_match", "math_verify", "response_length", "bleu",
            "rouge1", "rouge2", "rougeL", "bertscore", "meteor",
            "pass_at_k", "codebleu", "execution_accuracy".
            If None, benchmark defaults are used.
        temperature: Sampling temperature (0.0 = deterministic/greedy, 1.0 = standard,
            2.0 = very random). Recommended: 0.0 for evaluation reproducibility.
        max_tokens: Maximum tokens in model response. Typical values: 256 for short
            answers, 512 for medium, 2048 for long explanations or code.
        num_samples: Number of responses to generate per prompt. Use >1 for Pass@K
            metrics, ensembling, or measuring response variance.
        max_records_in_memory: Optional cap on generation/evaluation records kept in
            the returned report to bound memory for very large runs.
        workers: Number of parallel workers for generation. Higher = faster but may
            hit rate limits. Recommended: 4-16 for APIs, 32+ for local models.
        max_retries: Number of retries for generation failures (default: 3).
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
        >>> print(f"ExactMatch: {report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
        ExactMatch: 85.00%
    """
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
        raise ValueError(
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
            raise ValueError("`reference_field` is only supported for custom datasets.")
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
            raise ValueError(
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
            raise ValueError(
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
        raise TypeError("pipeline must implement EvaluationPipelineContract.")

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
    strategy_resolver = None
    if num_samples > 1:

        def strategy_resolver(task):  # noqa: ARG001
            return RepeatedSamplingStrategy(attempts=num_samples)

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
        raise ValueError(
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
        raise ValueError(
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
    dataset: object,
    *,
    cache_manager: CacheManager,
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
    raise ValueError(
        "No dataset provided. Supply `dataset=` to evaluate() "
        "or ensure `run_id` points to a cached run."
    )


def _resolve_storage(storage_path: str | Path | None, storage_backend: object | None):
    """Resolve storage backend from path or explicit backend."""
    if storage_backend is not None:
        backend = storage_backend
        if hasattr(backend, "experiment_storage"):
            return backend.experiment_storage
        if not hasattr(backend, "start_run"):
            raise TypeError("storage_backend must be ExperimentStorage-compatible.")
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
