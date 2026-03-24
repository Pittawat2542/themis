"""Curated public package surface for the current Themis runtime."""

from themis._version import __version__
from themis.benchmark import (
    BenchmarkDefinition,
    BenchmarkDefinitionConfig,
    BenchmarkSpec,
    DatasetQuerySpec,
    ParseSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    TraceScoreSpec,
    build_benchmark_definition_project,
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
from themis.specs.foundational import McpServerSpec, ModelSpec, ToolSpec

__all__ = [
    "__version__",
    "Orchestrator",
    "BenchmarkResult",
    "BenchmarkDefinition",
    "BenchmarkDefinitionConfig",
    "BenchmarkSpec",
    "SliceSpec",
    "DatasetQuerySpec",
    "PromptVariantSpec",
    "ParseSpec",
    "ScoreSpec",
    "TraceScoreSpec",
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
    "McpServerSpec",
    "ModelSpec",
    "ToolSpec",
    "EngineCapabilities",
    "PluginRegistry",
    "build_benchmark_definition_project",
    "generate_config_report",
]
