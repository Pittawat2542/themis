"""Catalog runtime registry assembly."""

from __future__ import annotations

from typing import cast

from themis.contracts.protocols import InferenceEngine, Metric
from themis import PluginRegistry
from themis.registry import EngineCapabilities

from .engines.common import DemoEngine, OpenAIChatEngine, OpenAICompatibleChatEngine
from .metrics.common import (
    ChoiceAccuracyMetric,
    ExactMatchMetric,
    MathEquivalenceMetric,
    NormalizedExactMatchMetric,
    NumericExactMatchMetric,
)
from .common import _normalize_provider_name

_SHARED_METRIC_FACTORIES: dict[str, type[Metric]] = {
    "exact_match": cast(type[Metric], ExactMatchMetric),
    "normalized_exact_match": cast(type[Metric], NormalizedExactMatchMetric),
    "choice_accuracy": cast(type[Metric], ChoiceAccuracyMetric),
    "numeric_exact_match": cast(type[Metric], NumericExactMatchMetric),
    "math_equivalence": cast(type[Metric], MathEquivalenceMetric),
}

_ENGINE_REGISTRATIONS: dict[str, tuple[type[InferenceEngine], EngineCapabilities]] = {
    "demo": (
        cast(type[InferenceEngine], DemoEngine),
        EngineCapabilities(supports_seed=True),
    ),
    "openai": (
        cast(type[InferenceEngine], OpenAIChatEngine),
        EngineCapabilities(supports_seed=True, supports_mcp=True),
    ),
    "openai_compatible": (
        cast(type[InferenceEngine], OpenAICompatibleChatEngine),
        EngineCapabilities(supports_seed=True),
    ),
}


def build_catalog_registry(providers: str | list[str]) -> PluginRegistry:
    """Build a registry containing the shared catalog metrics and requested engines."""

    registry = PluginRegistry.from_dict({"metrics": dict(_SHARED_METRIC_FACTORIES)})
    resolved_providers = [providers] if isinstance(providers, str) else list(providers)
    for provider in sorted(
        {_normalize_provider_name(provider) for provider in resolved_providers}
    ):
        register_catalog_engine(registry, provider)
    return registry


def register_catalog_metrics(registry: PluginRegistry) -> None:
    """Register the shared catalog metric set on an existing registry."""

    for name, factory in _SHARED_METRIC_FACTORIES.items():
        if not registry.has_metric(name):
            registry.register_metric(name, factory)


def register_catalog_engine(registry: PluginRegistry, provider: str) -> None:
    """Register one catalog inference engine on an existing registry."""

    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider in _ENGINE_REGISTRATIONS:
        factory, capabilities = _ENGINE_REGISTRATIONS[normalized_provider]
        registry.register_inference_engine(
            normalized_provider,
            factory,
            capabilities=capabilities,
        )
        return
    raise ValueError(f"Unsupported quick-start provider '{provider}'.")
