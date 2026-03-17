from __future__ import annotations

import themis


EXPECTED_ROOT_EXPORTS = {
    "__version__",
    "BenchmarkResult",
    "BenchmarkSpec",
    "DatasetQuerySpec",
    "ExecutionPolicySpec",
    "InferenceGridSpec",
    "InferenceParamsSpec",
    "ModelSpec",
    "Orchestrator",
    "ParseSpec",
    "PluginRegistry",
    "PostgresBlobStorageSpec",
    "ProjectSpec",
    "PromptMessage",
    "PromptVariantSpec",
    "ScoreSpec",
    "SliceSpec",
    "SqliteBlobStorageSpec",
    "StorageConfig",
    "StorageSpec",
    "generate_config_report",
}


def test_root_exports_only_benchmark_first_surface() -> None:
    assert set(themis.__all__) == EXPECTED_ROOT_EXPORTS
    assert not hasattr(themis, "ExperimentSpec")
    assert not hasattr(themis, "TaskSpec")
    assert not hasattr(themis, "PromptTemplateSpec")
    assert not hasattr(themis, "ItemSamplingSpec")
