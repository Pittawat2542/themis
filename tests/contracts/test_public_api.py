from __future__ import annotations

import themis
import themis.config_report
import themis.runtime
import themis.specs

from tests.constants import EXPECTED_ROOT_EXPORTS


def test_root_package_exports_only_benchmark_first_surface() -> None:
    assert EXPECTED_ROOT_EXPORTS.issubset(set(themis.__all__))
    assert "ExperimentSpec" not in themis.__all__
    assert "TaskSpec" not in themis.__all__
    assert "PromptTemplateSpec" not in themis.__all__
    assert "ItemSamplingSpec" not in themis.__all__
    assert "ExperimentResult" not in themis.__all__


def test_specs_namespace_exposes_public_benchmark_models() -> None:
    for name in [
        "BenchmarkSpec",
        "SliceSpec",
        "DatasetQuerySpec",
        "PromptVariantSpec",
        "ParseSpec",
        "ScoreSpec",
        "DatasetSpec",
        "GenerationSpec",
        "JudgeInferenceSpec",
    ]:
        assert hasattr(themis.specs, name)
        assert name in themis.specs.__all__

    for retired in [
        "ExperimentSpec",
        "TaskSpec",
        "PromptTemplateSpec",
        "ItemSamplingSpec",
        "OutputTransformSpec",
        "EvaluationSpec",
    ]:
        assert retired not in themis.specs.__all__


def test_runtime_namespace_exposes_benchmark_result_only() -> None:
    assert "BenchmarkResult" in themis.runtime.__all__
    assert "RecordTimelineView" in themis.runtime.__all__
    assert "ExperimentResult" not in themis.runtime.__all__
    assert "ComparisonTable" not in themis.runtime.__all__


def test_config_report_root_helper_remains_public() -> None:
    assert hasattr(themis, "generate_config_report")
    assert hasattr(themis.config_report, "generate_config_report")


def test_benchmark_result_public_methods_exist() -> None:
    result_type = themis.BenchmarkResult
    for method in ["aggregate", "paired_compare", "persist_artifacts", "report"]:
        assert hasattr(result_type, method)
    assert not hasattr(result_type, "build_report")
