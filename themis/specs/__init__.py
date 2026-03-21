"""Public benchmark-first spec models."""

from __future__ import annotations

from importlib import import_module

from themis.specs.base import SpecBase

__all__ = [
    "SpecBase",
    "BenchmarkSpec",
    "SliceSpec",
    "DatasetQuerySpec",
    "PromptVariantSpec",
    "ParseSpec",
    "ScoreSpec",
    "ModelSpec",
    "DatasetSpec",
    "GenerationSpec",
    "JudgeInferenceSpec",
    "StorageConfig",
    "StorageSpec",
    "SqliteBlobStorageSpec",
    "PostgresBlobStorageSpec",
    "ExecutionPolicySpec",
    "InferenceParamsSpec",
    "InferenceGridSpec",
    "PromptMessage",
    "PromptTurnSpec",
    "ProjectSpec",
]

_DYNAMIC_EXPORTS = {
    "BenchmarkSpec": ("themis.benchmark.specs", "BenchmarkSpec"),
    "SliceSpec": ("themis.benchmark.specs", "SliceSpec"),
    "DatasetQuerySpec": ("themis.benchmark.query", "DatasetQuerySpec"),
    "PromptVariantSpec": ("themis.benchmark.specs", "PromptVariantSpec"),
    "ParseSpec": ("themis.benchmark.specs", "ParseSpec"),
    "ScoreSpec": ("themis.benchmark.specs", "ScoreSpec"),
    "ModelSpec": ("themis.specs.foundational", "ModelSpec"),
    "DatasetSpec": ("themis.specs.foundational", "DatasetSpec"),
    "GenerationSpec": ("themis.specs.foundational", "GenerationSpec"),
    "JudgeInferenceSpec": ("themis.specs.foundational", "JudgeInferenceSpec"),
    "StorageConfig": ("themis.specs.experiment", "StorageConfig"),
    "StorageSpec": ("themis.specs.experiment", "StorageSpec"),
    "SqliteBlobStorageSpec": ("themis.specs.experiment", "SqliteBlobStorageSpec"),
    "PostgresBlobStorageSpec": ("themis.specs.experiment", "PostgresBlobStorageSpec"),
    "ExecutionPolicySpec": ("themis.specs.experiment", "ExecutionPolicySpec"),
    "InferenceParamsSpec": ("themis.specs.experiment", "InferenceParamsSpec"),
    "InferenceGridSpec": ("themis.specs.experiment", "InferenceGridSpec"),
    "PromptMessage": ("themis.specs.experiment", "PromptMessage"),
    "PromptTurnSpec": ("themis.specs.experiment", "PromptTurnSpec"),
    "ProjectSpec": ("themis.specs.experiment", "ProjectSpec"),
}


def __getattr__(name: str) -> object:
    if name not in _DYNAMIC_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _DYNAMIC_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
