"""Catalog runtime registry assembly."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

from themis.contracts.protocols import InferenceEngine, Metric, TrialMetric
from themis import PluginRegistry
from themis.registry import EngineCapabilities
from themis.registry.plugin_registry import MetricPlugin

from themis.catalog.metrics import (
    AccConsistencyMetric,
    AvgAtKMetric,
    BERTScoreMetric,
    BestOfNMetric,
    BleuMetric,
    ChrFMetric,
    EditDistanceMetric,
    MajorityAtKMetric,
    MeteorMetric,
    PassAtKMetric,
    RougeMetric,
    SacreBleuMetric,
    SelfConsistencyMetric,
    TERMetric,
    EventSequenceTraceMetric,
    NodePresenceTraceMetric,
    ToolPresenceTraceMetric,
    ToolStageTraceMetric,
    VarianceAtKMetric,
)
from .engines.common import DemoEngine, OpenAIChatEngine
from .metrics.common import (
    ChoiceAccuracyMetric,
    ExactMatchMetric,
    MathEquivalenceMetric,
    NormalizedExactMatchMetric,
    NumericExactMatchMetric,
)
from ._provider import _normalize_provider_name

MetricFactory = (
    Callable[[], MetricPlugin] | type[Metric] | type[TrialMetric] | MetricPlugin
)


_SHARED_METRIC_FACTORIES: dict[str, MetricFactory] = {
    "exact_match": cast(type[Metric], ExactMatchMetric),
    "normalized_exact_match": cast(type[Metric], NormalizedExactMatchMetric),
    "choice_accuracy": cast(type[Metric], ChoiceAccuracyMetric),
    "numeric_exact_match": cast(type[Metric], NumericExactMatchMetric),
    "math_equivalence": cast(type[Metric], MathEquivalenceMetric),
    "self_consistency": cast(type[Metric], SelfConsistencyMetric),
    "best_of_n": cast(type[Metric], BestOfNMetric),
    "pass_at_k": cast(type[Metric], PassAtKMetric),
    "avg_at_k": cast(type[Metric], AvgAtKMetric),
    "acc_consistency": cast(type[Metric], AccConsistencyMetric),
    "majority_at_k": cast(type[Metric], MajorityAtKMetric),
    "variance_at_k": cast(type[Metric], VarianceAtKMetric),
    "tool_presence": cast(type[Metric], ToolPresenceTraceMetric),
    "tool_stage": cast(type[Metric], ToolStageTraceMetric),
    "event_sequence": cast(type[Metric], EventSequenceTraceMetric),
    "node_presence": cast(type[Metric], NodePresenceTraceMetric),
    "bleu": cast(type[Metric], BleuMetric),
    "rouge_1": cast(type[Metric], partial(RougeMetric, "rouge1")),
    "rouge_2": cast(type[Metric], partial(RougeMetric, "rouge2")),
    "rouge_l": cast(type[Metric], partial(RougeMetric, "rougeL")),
    "meteor": cast(type[Metric], MeteorMetric),
    "bertscore": cast(type[Metric], BERTScoreMetric),
    "sacrebleu": cast(type[Metric], SacreBleuMetric),
    "chrf": cast(type[Metric], ChrFMetric),
    "ter": cast(type[Metric], TERMetric),
    "edit_distance": cast(type[Metric], EditDistanceMetric),
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
