from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from themis import (
    BenchmarkDefinition,
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.records import MetricScore
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import DatasetSource, PromptRole
from themis.types.json_types import JSONDict


class _PreviewReadyDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return []

    def prepare_rows(self, rows, dataset):
        del dataset
        return type("PreparedRows", (), {"rows": [{"input": rows[0]["question"]}]})()


class _DummyMetric:
    def score(self, trial, candidate, context) -> MetricScore:
        del trial, candidate, context
        return MetricScore(metric_id="dummy_metric", value=1.0)


def _build_definition(*, requires_judge: bool = False) -> BenchmarkDefinition:
    def _builder(definition: BenchmarkDefinition, config) -> BenchmarkSpec:
        del definition
        return BenchmarkSpec(
            benchmark_id="demo-definition",
            models=[ModelSpec(model_id=config.model_id, provider=config.provider)],
            slices=[
                SliceSpec(
                    slice_id="demo-slice",
                    dataset=DatasetSpec(
                        source=DatasetSource.MEMORY,
                        dataset_id="memory://demo",
                    ),
                    dataset_query=DatasetQuerySpec(),
                    prompt_variant_ids=["demo-default"],
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="demo-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="{item.input}",
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        )

    def _registrar(
        definition: BenchmarkDefinition,
        registry: PluginRegistry,
        config,
    ) -> None:
        del definition, config
        registry.register_metric("demo_metric", _DummyMetric)

    def _summarizer(definition: BenchmarkDefinition, result: object) -> JSONDict:
        del result
        return {
            "benchmark_id": definition.benchmark_id,
            "result": {"status": "ok"},
        }

    return BenchmarkDefinition(
        benchmark_id="demo-definition",
        family="demo",
        primary_metric_id="exact_match",
        requires_judge=requires_judge,
        metadata={"dataset_id": "demo/dataset", "split": "test"},
        builder=_builder,
        registrar=_registrar,
        summarizer=_summarizer,
        dataset_provider_factory=lambda definition, **kwargs: (
            _PreviewReadyDatasetProvider()
        ),
        preview_rows_loader=lambda definition: [{"question": "What is 2 + 2?"}],
    )


def test_benchmark_definition_runtime_config_normalizes_providers_and_requires_judge() -> (
    None
):
    definition = _build_definition(requires_judge=True)

    with pytest.raises(ValueError, match="judge"):
        definition.build_runtime_config(model_id="demo-model", provider="openai-demo")

    config = definition.build_runtime_config(
        model_id="demo-model",
        provider="openai",
        judge_model_id="judge-model",
        judge_provider="demo-provider",
    )

    assert config.provider == "openai"
    assert config.judge_provider == "demo_provider"


def test_benchmark_definition_registers_summarizes_and_renders_preview() -> None:
    definition = _build_definition()
    registry = PluginRegistry()

    definition.register_required_components(registry)
    summary = definition.summarize_result({"score": 1.0})
    preview = definition.render_preview(model_id="demo-model", provider="demo")

    assert registry.has_metric("demo_metric")
    assert summary["benchmark_id"] == "demo-definition"
    preview_entry = cast(dict[str, object], preview[0])
    messages = cast(list[dict[str, object]], preview_entry["messages"])
    assert messages[0]["content"] == "What is 2 + 2?"


def test_build_benchmark_definition_project_builds_project_bundle(
    tmp_path: Path,
) -> None:
    from themis.benchmark import build_benchmark_definition_project

    definition = _build_definition()

    def _build_registry(providers: str | list[str]) -> PluginRegistry:
        del providers
        registry = PluginRegistry()
        registry.register_metric("seed_metric", _DummyMetric)
        return registry

    project, benchmark, registry, dataset_provider, resolved_definition = (
        build_benchmark_definition_project(
            definition=definition,
            model_id="demo-model",
            provider="demo-provider",
            storage_root=tmp_path / "bundle-store",
            build_registry=_build_registry,
        )
    )

    assert isinstance(project, ProjectSpec)
    assert project.execution_policy == ExecutionPolicySpec()
    assert isinstance(benchmark, BenchmarkSpec)
    assert registry.has_metric("seed_metric")
    assert dataset_provider is not None
    assert resolved_definition is definition
