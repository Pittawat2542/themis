"""Helper functions for the Themis API evaluation surface.

These functions encapsulate the complex configuration and wiring logic required
to prepare an ExperimentSession from the high-level `evaluate()` inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from themis.exceptions import ConfigurationError
from themis.interfaces import DatasetAdapter, Extractor, Metric
from themis.generation.templates import PromptTemplate
from themis.providers.options import normalize_provider_options

if TYPE_CHECKING:
    from themis.evaluation.pipeline import EvaluationPipelineContract
    from themis.experiment.cache_manager import CacheManager

logger = logging.getLogger(__name__)

__all__ = [
    "_build_evaluation_config",
    "_dataset_fingerprint",
    "_extract_provider_options",
    "_metrics_require_references",
    "_prompt_fingerprint",
    "_resolve_custom_reference_field",
    "_resolve_dataset_with_cache",
    "_resolve_evaluation_context",
    "_should_warn_missing_api_key",
    "_ALLOWED_EXTRA_OPTIONS",
    "_EvaluationContext",
]


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


@dataclass
class _EvaluationContext:
    dataset: Sequence[dict[str, Any]]
    prompt_template: PromptTemplate
    metrics_list: list[Metric]
    extractor: Extractor
    metadata_fields: tuple[str, ...]
    reference_field: str | None
    dataset_id_field: str


def _resolve_evaluation_context(
    benchmark_or_dataset: str | Sequence[dict[str, Any]],
    limit: int | None,
    prompt: str | None,
    reference_field: str | None,
    metrics: list[str] | None,
) -> _EvaluationContext:
    is_benchmark = isinstance(benchmark_or_dataset, str)

    if is_benchmark:
        if reference_field is not None:
            raise ConfigurationError(
                "`reference_field` is only supported for custom datasets."
            )
        benchmark_name = benchmark_or_dataset
        logger.info(f"Loading benchmark: {benchmark_name}")

        from themis.presets import get_benchmark_preset

        try:
            preset = get_benchmark_preset(benchmark_or_dataset)  # type: ignore
        except Exception as e:
            logger.error(f"❌ Failed to get benchmark preset '{benchmark_name}': {e}")
            raise

        logger.info(f"Loading dataset (limit={limit})...")
        try:
            dataset = preset.load_dataset(limit=limit)
            logger.info(f"✅ Loaded {len(dataset)} samples from {benchmark_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise

        if prompt is None:
            prompt_template = preset.prompt_template  # type: ignore
        else:
            prompt_template = PromptTemplate(name="custom", template=prompt)

        from themis.evaluation.metric_resolver import resolve_metrics

        if metrics is None:
            metrics_list = preset.metrics
        else:
            metrics_list = resolve_metrics(metrics)

        extractor = preset.extractor
        metadata_fields = ()
        selected_reference_field = preset.reference_field
        dataset_id_field = preset.dataset_id_field

    else:
        logger.info("Using custom dataset")
        dataset = list(benchmark_or_dataset)  # type: ignore
        logger.info(f"Custom dataset has {len(dataset)} samples")

        if limit is not None:
            dataset = dataset[:limit]
            logger.info(f"Limited to {len(dataset)} samples")

        if prompt is None:
            raise ConfigurationError(
                "Custom datasets require a prompt template. "
                "Example: prompt='Solve: {question}'"
            )
        prompt_template = PromptTemplate(name="custom", template=prompt)

        from themis.evaluation.metric_resolver import resolve_metrics

        if metrics is None:
            metrics_list = resolve_metrics(["exact_match"])
        else:
            metrics_list = resolve_metrics(metrics)

        from themis.evaluation.extractors import IdentityExtractor

        extractor = IdentityExtractor()

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

    return _EvaluationContext(
        dataset=dataset,
        prompt_template=prompt_template,
        metrics_list=metrics_list,
        extractor=extractor,
        metadata_fields=metadata_fields,
        reference_field=selected_reference_field,
        dataset_id_field=dataset_id_field,
    )
