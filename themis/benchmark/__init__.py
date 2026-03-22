"""Public benchmark-first authoring surface."""

from themis.benchmark.compiler import compile_benchmark
from themis.benchmark.definitions import (
    BenchmarkDefinition,
    BenchmarkDefinitionConfig,
    build_benchmark_definition_project,
)
from themis.benchmark.query import DatasetQuerySpec
from themis.benchmark.specs import (
    BenchmarkSpec,
    DatasetSliceSpec,
    ParseSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)

__all__ = [
    "BenchmarkDefinition",
    "BenchmarkDefinitionConfig",
    "BenchmarkSpec",
    "DatasetSliceSpec",
    "DatasetQuerySpec",
    "ParseSpec",
    "PromptVariantSpec",
    "ScoreSpec",
    "SliceSpec",
    "build_benchmark_definition_project",
    "compile_benchmark",
]
