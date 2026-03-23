"""Reusable preset-definition layer above BenchmarkSpec."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from themis.benchmark.specs import BenchmarkSpec
from themis.contracts.protocols import DatasetProvider
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import (
    ExecutionPolicySpec,
    ProjectSpec,
    SqliteBlobStorageSpec,
)
from themis.types.enums import CompressionCodec
from themis.types.json_types import JSONDict

BenchmarkRow = dict[str, object]

BenchmarkBuilder = Callable[
    ["BenchmarkDefinition", "BenchmarkDefinitionConfig"],
    BenchmarkSpec,
]
BenchmarkRegistrar = Callable[
    ["BenchmarkDefinition", PluginRegistry, "BenchmarkDefinitionConfig"],
    None,
]
BenchmarkSummarizer = Callable[["BenchmarkDefinition", object], JSONDict]
DatasetProviderFactory = Callable[..., DatasetProvider]
PreviewRenderer = Callable[
    ["BenchmarkDefinition", "BenchmarkDefinitionConfig", dict[str, object]],
    list[JSONDict],
]
PreviewRowsLoader = Callable[["BenchmarkDefinition"], list[BenchmarkRow]]
RegistryBuilder = Callable[[str | list[str]], PluginRegistry]


@dataclass(frozen=True, slots=True)
class BenchmarkDefinitionConfig:
    model_id: str
    provider: str
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    num_samples: int = 1
    dataset_revision: str | None = None
    subset: int | None = None
    judge_model_id: str | None = None
    judge_provider: str | None = None


@dataclass(slots=True)
class BenchmarkDefinition:
    benchmark_id: str
    family: str
    primary_metric_id: str | None
    requires_judge: bool
    metadata: JSONDict
    builder: BenchmarkBuilder
    registrar: BenchmarkRegistrar
    summarizer: BenchmarkSummarizer
    dataset_provider_factory: DatasetProviderFactory | None = None
    preview_renderer: PreviewRenderer | None = None
    preview_rows_loader: PreviewRowsLoader | None = None

    def build_runtime_config(
        self,
        *,
        model_id: str,
        provider: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        num_samples: int = 1,
        dataset_revision: str | None = None,
        subset: int | None = None,
        judge_model_id: str | None = None,
        judge_provider: str | None = None,
    ) -> BenchmarkDefinitionConfig:
        if self.requires_judge and (not judge_model_id or not judge_provider):
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' requires explicit "
                "judge_model_id and judge_provider."
            )
        return BenchmarkDefinitionConfig(
            model_id=model_id,
            provider=_normalize_provider_name(provider),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            num_samples=num_samples,
            dataset_revision=dataset_revision,
            subset=subset,
            judge_model_id=judge_model_id,
            judge_provider=(
                _normalize_provider_name(judge_provider)
                if judge_provider is not None
                else None
            ),
        )

    def build_benchmark(
        self,
        *,
        model_id: str,
        provider: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        num_samples: int = 1,
        dataset_revision: str | None = None,
        subset: int | None = None,
        judge_model_id: str | None = None,
        judge_provider: str | None = None,
    ) -> BenchmarkSpec:
        config = self.build_runtime_config(
            model_id=model_id,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            num_samples=num_samples,
            dataset_revision=dataset_revision,
            subset=subset,
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        return self.builder(self, config).model_copy(
            update={"num_samples": config.num_samples}
        )

    def register_required_components(
        self,
        registry: PluginRegistry,
        *,
        judge_model_id: str | None = None,
        judge_provider: str | None = None,
    ) -> None:
        config = self.build_runtime_config(
            model_id="preview-model",
            provider="demo",
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        self.registrar(self, registry, config)

    def build_dataset_provider(
        self,
        *,
        huggingface_loader=None,
    ) -> DatasetProvider:
        if self.dataset_provider_factory is None:
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' does not define a dataset provider."
            )
        return self.dataset_provider_factory(
            self, huggingface_loader=huggingface_loader
        )

    def render_preview(
        self,
        *,
        model_id: str = "preview-model",
        provider: str = "demo",
        judge_model_id: str | None = None,
        judge_provider: str | None = None,
    ) -> list[JSONDict]:
        if self.preview_rows_loader is None:
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' does not define preview rows."
            )
        config = self.build_runtime_config(
            model_id=model_id,
            provider=provider,
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        benchmark = self.builder(self, config)
        sample_rows = self.preview_rows_loader(self)
        if not sample_rows:
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' preview rows are empty."
            )
        preview_slice = benchmark.slices[0]
        provider_instance = self.build_dataset_provider()
        if hasattr(provider_instance, "prepare_rows"):
            prepared = provider_instance.prepare_rows(sample_rows, preview_slice)
            sample_rows = cast(list[BenchmarkRow], prepared.rows)
        if not sample_rows:
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' preview rows are empty."
            )
        sample = sample_rows[0]
        if self.preview_renderer is not None:
            return self.preview_renderer(self, config, sample)
        return benchmark.preview(sample)

    def summarize_result(self, result) -> JSONDict:
        return self.summarizer(self, result)


def build_benchmark_definition_project(
    *,
    definition: BenchmarkDefinition,
    model_id: str,
    provider: str,
    storage_root: Path,
    build_registry: RegistryBuilder,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = None,
    num_samples: int = 1,
    subset: int | None = None,
    dataset_revision: str | None = None,
    judge_model_id: str | None = None,
    judge_provider: str | None = None,
    huggingface_loader=None,
) -> tuple[
    ProjectSpec,
    BenchmarkSpec,
    PluginRegistry,
    DatasetProvider,
    BenchmarkDefinition,
]:
    benchmark = definition.build_benchmark(
        model_id=model_id,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        num_samples=num_samples,
        subset=subset,
        dataset_revision=dataset_revision,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
    )
    providers: list[str] = [provider]
    if judge_provider is not None:
        providers.append(judge_provider)
    registry = build_registry(providers)
    definition.register_required_components(
        registry,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
    )
    project = ProjectSpec(
        project_name=f"quick-eval-{definition.benchmark_id}",
        researcher_id="themis-cli",
        global_seed=seed or 7,
        storage=SqliteBlobStorageSpec(
            root_dir=str(storage_root),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    provider_instance = definition.build_dataset_provider(
        huggingface_loader=huggingface_loader
    )
    return project, benchmark, registry, provider_instance, definition


def _normalize_provider_name(provider: str) -> str:
    return provider.replace("-", "_")
