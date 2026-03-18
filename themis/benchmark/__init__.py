"""Public benchmark-first authoring surface."""

from themis.benchmark.compiler import compile_benchmark
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
    "BenchmarkSpec",
    "DatasetSliceSpec",
    "DatasetQuerySpec",
    "ParseSpec",
    "PromptVariantSpec",
    "ScoreSpec",
    "SliceSpec",
    "compile_benchmark",
]
