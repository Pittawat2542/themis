"""Curated public package surface for the current Themis runtime."""

from themis._version import __version__
from themis.benchmark import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ParseSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.config_report import generate_config_report
from themis.orchestration.orchestrator import Orchestrator
from themis.runtime import BenchmarkResult
from themis.specs.experiment import (
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    PostgresBlobStorageSpec,
    PromptMessage,
    PromptTurnSpec,
    ProjectSpec,
    SqliteBlobStorageSpec,
    StorageConfig,
    StorageSpec,
)
from themis.registry.plugin_registry import EngineCapabilities, PluginRegistry
from themis.specs.foundational import ModelSpec, ToolSpec

__all__ = [
    "__version__",
    "Orchestrator",
    "BenchmarkResult",
    "BenchmarkSpec",
    "SliceSpec",
    "DatasetQuerySpec",
    "PromptVariantSpec",
    "ParseSpec",
    "ScoreSpec",
    "ProjectSpec",
    "StorageConfig",
    "StorageSpec",
    "SqliteBlobStorageSpec",
    "PostgresBlobStorageSpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "PromptMessage",
    "PromptTurnSpec",
    "InferenceParamsSpec",
    "ModelSpec",
    "ToolSpec",
    "EngineCapabilities",
    "PluginRegistry",
    "generate_config_report",
]
