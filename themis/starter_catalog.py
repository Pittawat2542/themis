"""Public starter helpers and built-in benchmark catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
import json
import math
from pathlib import Path
from typing import cast

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    ParseSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    SqliteBlobStorageSpec,
)
from themis._optional import import_optional
from themis.specs.foundational import DatasetSpec, ExtractorRefSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole
from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict

from themis._starter_catalog import datasets as _datasets
from themis._starter_catalog import runtime as _runtime
from themis._starter_catalog.datasets import (
    BuiltinEncycloKDatasetProvider,
    BuiltinHealthBenchDatasetProvider,
    BuiltinHLEDatasetProvider,
    BuiltinLPFQADatasetProvider,
    BuiltinMMLUProDatasetProvider,
    BuiltinSimpleQAVerifiedDatasetProvider,
    BuiltinSuperGPQADatasetProvider,
    StarterDatasetProvider,
    StarterNormalizedRows,
    _prompt_messages_from_context,
)
from themis._starter_catalog.runtime import (
    ChoiceAccuracyMetric,
    HLEJudgeMetric,
    HealthBenchRubricMetric,
    LPFQAJudgeMetric,
    SimpleQAVerifiedJudgeMetric,
    _normalize_provider_name,
    _provider_model_extras,
    build_starter_registry,
    register_starter_engine,
    register_starter_metrics,
)

_MCQ_CHOICES = list("ABCDEFGHIJ")

BenchmarkBuilder = Callable[
    ["BuiltinBenchmarkDefinition", "BuiltinBenchmarkRuntimeConfig"],
    BenchmarkSpec,
]
BenchmarkRegistrar = Callable[
    ["BuiltinBenchmarkDefinition", PluginRegistry, "BuiltinBenchmarkRuntimeConfig"],
    None,
]
BenchmarkSummarizer = Callable[["BuiltinBenchmarkDefinition", object], JSONDict]
DatasetProviderFactory = Callable[..., StarterDatasetProvider]
PreviewRenderer = Callable[
    ["BuiltinBenchmarkDefinition", "BuiltinBenchmarkRuntimeConfig", dict[str, object]],
    list[JSONDict],
]

_build_judge_spec = _runtime._build_judge_spec
_normalize_healthbench_rows = _datasets._normalize_healthbench_rows


@dataclass(frozen=True, slots=True)
class BuiltinBenchmarkRuntimeConfig:
    model_id: str
    provider: str
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    dataset_revision: str | None = None
    subset: int | None = None
    judge_model_id: str | None = None
    judge_provider: str | None = None


@dataclass(slots=True)
class BuiltinBenchmarkDefinition:
    benchmark_id: str
    dataset_id: str
    split: str
    metric_id: str
    requires_judge: bool
    builder: BenchmarkBuilder
    registrar: BenchmarkRegistrar
    summarizer: BenchmarkSummarizer
    dataset_provider_factory: DatasetProviderFactory | None = None
    preview_renderer: PreviewRenderer | None = None

    def build_runtime_config(
        self,
        *,
        model_id: str,
        provider: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        dataset_revision: str | None = None,
        subset: int | None = None,
        judge_model_id: str | None = None,
        judge_provider: str | None = None,
    ) -> BuiltinBenchmarkRuntimeConfig:
        if self.requires_judge and (not judge_model_id or not judge_provider):
            raise ValueError(
                f"Built-in benchmark '{self.benchmark_id}' requires explicit "
                "judge_model_id and judge_provider."
            )
        return BuiltinBenchmarkRuntimeConfig(
            model_id=model_id,
            provider=_normalize_provider_name(provider),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
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
            dataset_revision=dataset_revision,
            subset=subset,
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        return self.builder(self, config)

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
    ) -> StarterDatasetProvider:
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
        config = self.build_runtime_config(
            model_id=model_id,
            provider=provider,
            judge_model_id=judge_model_id,
            judge_provider=judge_provider,
        )
        fixture = _load_fixture(self.benchmark_id)
        benchmark = self.builder(self, config)
        dataset = benchmark.slices[0].dataset
        provider_instance = self.build_dataset_provider()
        sample_rows = provider_instance.prepare_rows(fixture["samples"], dataset).rows
        sample = sample_rows[0]
        if self.preview_renderer is not None:
            return self.preview_renderer(self, config, sample)
        return benchmark.preview(sample)

    def summarize_result(self, result) -> JSONDict:
        return self.summarizer(self, result)


def load_local_rows(path: Path) -> list[dict[str, object]]:
    return _datasets.load_local_rows(path)


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
) -> list[dict[str, object]]:
    datasets_module = import_optional("datasets", extra="datasets")
    return _datasets.load_huggingface_rows(
        dataset_id,
        split,
        revision,
        datasets_module=datasets_module,
    )


def inspect_huggingface_dataset(
    dataset_id: str,
    *,
    split: str = "test",
    revision: str | None = None,
    metadata_loader=None,
    row_loader=None,
    max_samples: int = 3,
) -> JSONDict:
    datasets_module = None
    if metadata_loader is None or row_loader is None:
        datasets_module = import_optional("datasets", extra="datasets")
    return _datasets.inspect_huggingface_dataset(
        dataset_id,
        split=split,
        revision=revision,
        metadata_loader=metadata_loader,
        row_loader=row_loader,
        max_samples=max_samples,
        datasets_module=datasets_module,
    )


def build_builtin_benchmark_project(
    *,
    benchmark_id: str,
    model_id: str,
    provider: str,
    storage_root: Path,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = None,
    subset: int | None = None,
    dataset_revision: str | None = None,
    judge_model_id: str | None = None,
    judge_provider: str | None = None,
    huggingface_loader=None,
) -> tuple[
    ProjectSpec,
    BenchmarkSpec,
    PluginRegistry,
    StarterDatasetProvider,
    BuiltinBenchmarkDefinition,
]:
    definition = get_builtin_benchmark(benchmark_id)
    benchmark = definition.build_benchmark(
        model_id=model_id,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        subset=subset,
        dataset_revision=dataset_revision,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
    )
    providers = [provider]
    if judge_provider is not None:
        providers.append(judge_provider)
    registry = build_starter_registry(providers)
    definition.register_required_components(
        registry,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
    )
    project = ProjectSpec(
        project_name=f"quick-eval-{benchmark_id}",
        researcher_id="themis-cli",
        global_seed=seed or 7,
        storage=SqliteBlobStorageSpec(
            root_dir=str(storage_root),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    provider_instance = definition.build_dataset_provider(
        huggingface_loader=load_huggingface_rows
        if huggingface_loader is None
        else huggingface_loader
    )
    return project, benchmark, registry, provider_instance, definition


def list_builtin_benchmarks() -> list[str]:
    return sorted(_BUILTIN_BENCHMARKS)


def get_builtin_benchmark(name: str) -> BuiltinBenchmarkDefinition:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in _BUILTIN_BENCHMARKS:
        raise ValueError(
            "Unknown built-in benchmark. Choose one of: "
            + ", ".join(sorted(_BUILTIN_BENCHMARKS))
        )
    return _BUILTIN_BENCHMARKS[normalized]


def _iter_score_rows(result, metric_id: str) -> list[ScoreRow]:
    return list(
        result.projection_repo.iter_candidate_scores(
            trial_hashes=result.trial_hashes,
            metric_id=metric_id,
            evaluation_hash=getattr(result, "active_evaluation_hash", None),
        )
    )


def _mean_summary(metric_id: str, result) -> JSONDict:
    rows = _iter_score_rows(result, metric_id)
    count = len(rows)
    mean = sum(row.score for row in rows) / count if count else 0.0
    return {"metric_id": metric_id, "count": count, "mean": mean}


def _summarize_simpleqa(_definition, result) -> JSONDict:
    rows = _iter_score_rows(result, "simpleqa_verified_score")
    count = len(rows)
    if count == 0:
        return {
            "metric_id": "simpleqa_verified_score",
            "count": 0,
            "correct_rate": 0.0,
            "attempted_rate": 0.0,
            "accuracy_given_attempted": 0.0,
            "f1": 0.0,
        }
    correct_rate = (
        sum(1.0 for row in rows if row.details.get("grade") == "CORRECT") / count
    )
    incorrect_rate = (
        sum(1.0 for row in rows if row.details.get("grade") == "INCORRECT") / count
    )
    attempted_rate = correct_rate + incorrect_rate
    accuracy_given_attempted = (
        correct_rate / attempted_rate if attempted_rate > 0 else 0.0
    )
    f1 = (
        2
        * accuracy_given_attempted
        * correct_rate
        / (accuracy_given_attempted + correct_rate)
        if (accuracy_given_attempted + correct_rate) > 0
        else 0.0
    )
    return {
        "metric_id": "simpleqa_verified_score",
        "count": count,
        "correct_rate": correct_rate,
        "attempted_rate": attempted_rate,
        "accuracy_given_attempted": accuracy_given_attempted,
        "f1": f1,
    }


def _summarize_healthbench(_definition, result) -> JSONDict:
    rows = _iter_score_rows(result, "healthbench_score")
    count = len(rows)
    mean_score = sum(row.score for row in rows) / count if count else 0.0
    tag_values: dict[str, list[float]] = {}
    for row in rows:
        for tag in row.details.get("example_tags", []):
            if isinstance(tag, str):
                tag_values.setdefault(tag, []).append(row.score)
    return {
        "metric_id": "healthbench_score",
        "count": count,
        "mean_overall_score": mean_score,
        "tag_means": {
            tag: sum(values) / len(values) for tag, values in sorted(tag_values.items())
        },
    }


def _summarize_hle(_definition, result) -> JSONDict:
    rows = _iter_score_rows(result, "hle_accuracy")
    count = len(rows)
    accuracy = sum(row.score for row in rows) / count if count else 0.0
    confidence_interval_half_width = (
        1.96 * math.sqrt(accuracy * (1 - accuracy) / count) if count else 0.0
    )
    confidences = [
        max(0.0, min(1.0, float(row.details.get("confidence", 100)) / 100.0))
        for row in rows
    ]
    truths = [float(bool(row.details.get("correct", False))) for row in rows]
    calibration_error = _hle_calibration_error(
        confidences=confidences,
        truths=truths,
        accuracy=accuracy,
    )
    scan_stats = getattr(result, "_builtin_scan_stats", {}) or {}
    return {
        "metric_id": "hle_accuracy",
        "count": count,
        "accuracy": accuracy,
        "confidence_interval_half_width": confidence_interval_half_width,
        "calibration_error": calibration_error,
        "skipped_image_count": int(scan_stats.get("skipped_image_count", 0) or 0),
    }


def _hle_calibration_error(
    *,
    confidences: list[float],
    truths: list[float],
    accuracy: float,
) -> float:
    if not confidences:
        return 0.0
    if len(confidences) < 100:
        incorrect_deltas = [
            abs(confidence - accuracy)
            for confidence, truth in zip(confidences, truths, strict=True)
            if truth < 1.0
        ]
        if incorrect_deltas:
            return sum(incorrect_deltas) / len(incorrect_deltas)
        return abs(sum(confidences) / len(confidences) - accuracy)
    paired = sorted(zip(confidences, truths, strict=True), key=lambda pair: pair[0])
    beta = 100
    calibration = 0.0
    total_examples = len(paired)
    for start in range(0, total_examples, beta):
        bucket = paired[start : start + beta]
        if not bucket:
            continue
        bucket_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        bucket_accuracy = sum(truth for _, truth in bucket) / len(bucket)
        difference = abs(bucket_confidence - bucket_accuracy)
        calibration += len(bucket) / total_examples * (difference**2)
    return math.sqrt(calibration)


def _summarize_mcq(definition, result) -> JSONDict:
    return _mean_summary(definition.metric_id, result)


def _summarize_lpfqa(definition, result) -> JSONDict:
    return _mean_summary(definition.metric_id, result)


def _make_dataset_query(config: BuiltinBenchmarkRuntimeConfig) -> DatasetQuerySpec:
    if config.subset is None:
        return DatasetQuerySpec()
    return DatasetQuerySpec.subset(config.subset, seed=config.seed)


def _mcq_dataset_spec(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
    *,
    expected_source_field: str,
) -> DatasetSpec:
    return DatasetSpec(
        source=DatasetSource.HUGGINGFACE,
        dataset_id=definition.dataset_id,
        split=definition.split,
        revision=config.dataset_revision,
        transforms=[
            {
                "kind": "rename",
                "field": "expected",
                "source_field": expected_source_field,
            },
            {
                "kind": "jinja",
                "field": "prompt_text",
                "template": (
                    "Question:\n{question}\n\nOptions:\n{options_text}\n\n"
                    "Return the best option letter only."
                ),
            },
        ],
    )


def _build_mcq_benchmark(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
    *,
    expected_source_field: str,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=_mcq_dataset_spec(
                    definition,
                    config,
                    expected_source_field=expected_source_field,
                ),
                dataset_query=_make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                parses=[
                    ParseSpec(
                        name="parsed",
                        extractors=[
                            ExtractorRefSpec(
                                id="choice_letter",
                                config={"choices": _MCQ_CHOICES},
                            )
                        ],
                    )
                ],
                scores=[
                    ScoreSpec(
                        name="default",
                        parse="parsed",
                        metrics=["choice_accuracy"],
                    )
                ],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.prompt_text}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _build_simpleqa_benchmark(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=definition.dataset_id,
                    split=definition.split,
                    revision=config.dataset_revision,
                    transforms=[
                        {
                            "kind": "rename",
                            "field": "prompt_text",
                            "source_field": "problem",
                        }
                    ],
                ),
                dataset_query=_make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["simpleqa_verified_score"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.prompt_text}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _build_healthbench_benchmark(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=definition.dataset_id,
                    split=definition.split,
                    revision=config.dataset_revision,
                ),
                dataset_query=_make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["healthbench_score"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.prompt_text}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _build_lpfqa_benchmark(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=definition.dataset_id,
                    split=definition.split,
                    revision=config.dataset_revision,
                    transforms=[
                        {
                            "kind": "rename",
                            "field": "prompt_text",
                            "source_field": "prompt",
                        }
                    ],
                ),
                dataset_query=_make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["lpfqa_score"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.prompt_text}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _build_hle_benchmark(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=definition.dataset_id,
                    split=definition.split,
                    revision=config.dataset_revision,
                    transforms=[
                        {
                            "kind": "rename",
                            "field": "expected",
                            "source_field": "answer",
                        },
                        {
                            "kind": "jinja",
                            "field": "prompt_text",
                            "template": (
                                "Your response should be in the following format:\n"
                                "Explanation: your explanation for your answer choice\n"
                                "Answer: your chosen answer\n"
                                "Confidence: your confidence score between 0% and 100% "
                                "for your answer\n\nQuestion:\n{question}"
                            ),
                        },
                    ],
                ),
                dataset_query=_make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["hle_accuracy"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.prompt_text}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[
                InferenceParamsSpec(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                )
            ]
        ),
    )


def _register_mcq(
    _definition,
    registry: PluginRegistry,
    config: BuiltinBenchmarkRuntimeConfig,
) -> None:
    del config
    if not registry.has_metric("choice_accuracy"):
        registry.register_metric("choice_accuracy", ChoiceAccuracyMetric())


def _register_simpleqa(
    _definition,
    registry: PluginRegistry,
    config: BuiltinBenchmarkRuntimeConfig,
) -> None:
    registry.register_metric(
        "simpleqa_verified_score",
        SimpleQAVerifiedJudgeMetric(
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def _register_healthbench(
    _definition,
    registry: PluginRegistry,
    config: BuiltinBenchmarkRuntimeConfig,
) -> None:
    registry.register_metric(
        "healthbench_score",
        HealthBenchRubricMetric(
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def _register_lpfqa(
    _definition,
    registry: PluginRegistry,
    config: BuiltinBenchmarkRuntimeConfig,
) -> None:
    registry.register_metric(
        "lpfqa_score",
        LPFQAJudgeMetric(
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def _register_hle(
    _definition,
    registry: PluginRegistry,
    config: BuiltinBenchmarkRuntimeConfig,
) -> None:
    registry.register_metric(
        "hle_accuracy",
        HLEJudgeMetric(
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def _render_healthbench_preview(
    definition: BuiltinBenchmarkDefinition,
    config: BuiltinBenchmarkRuntimeConfig,
    sample: dict[str, object],
) -> list[JSONDict]:
    del definition, config
    return [
        {
            "prompt_variant_id": "healthbench-default",
            "messages": _prompt_messages_from_context(sample),
            "follow_up_turns": [],
        }
    ]


def _load_fixture(benchmark_id: str) -> JSONDict:
    with (
        resources.files("themis")
        .joinpath("starter_fixtures", f"{benchmark_id}.json")
        .open("r", encoding="utf-8") as fh
    ):
        return cast(JSONDict, json.load(fh))


_BUILTIN_BENCHMARKS: dict[str, BuiltinBenchmarkDefinition] = {
    "mmlu_pro": BuiltinBenchmarkDefinition(
        benchmark_id="mmlu_pro",
        dataset_id="TIGER-Lab/MMLU-Pro",
        split="test",
        metric_id="choice_accuracy",
        requires_judge=False,
        builder=lambda definition, config: _build_mcq_benchmark(
            definition,
            config,
            expected_source_field="answer",
        ),
        registrar=_register_mcq,
        summarizer=_summarize_mcq,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinMMLUProDatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
    "supergpqa": BuiltinBenchmarkDefinition(
        benchmark_id="supergpqa",
        dataset_id="m-a-p/SuperGPQA",
        split="train",
        metric_id="choice_accuracy",
        requires_judge=False,
        builder=lambda definition, config: _build_mcq_benchmark(
            definition,
            config,
            expected_source_field="answer_letter",
        ),
        registrar=_register_mcq,
        summarizer=_summarize_mcq,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinSuperGPQADatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
    "encyclo_k": BuiltinBenchmarkDefinition(
        benchmark_id="encyclo_k",
        dataset_id="m-a-p/Encyclo-K",
        split="test",
        metric_id="choice_accuracy",
        requires_judge=False,
        builder=lambda definition, config: _build_mcq_benchmark(
            definition,
            config,
            expected_source_field="answer_letter",
        ),
        registrar=_register_mcq,
        summarizer=_summarize_mcq,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinEncycloKDatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
    "simpleqa_verified": BuiltinBenchmarkDefinition(
        benchmark_id="simpleqa_verified",
        dataset_id="google/simpleqa-verified",
        split="eval",
        metric_id="simpleqa_verified_score",
        requires_judge=True,
        builder=_build_simpleqa_benchmark,
        registrar=_register_simpleqa,
        summarizer=_summarize_simpleqa,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinSimpleQAVerifiedDatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
    "healthbench": BuiltinBenchmarkDefinition(
        benchmark_id="healthbench",
        dataset_id="openai/healthbench",
        split="test",
        metric_id="healthbench_score",
        requires_judge=True,
        builder=_build_healthbench_benchmark,
        registrar=_register_healthbench,
        summarizer=_summarize_healthbench,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinHealthBenchDatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
        preview_renderer=_render_healthbench_preview,
    ),
    "lpfqa": BuiltinBenchmarkDefinition(
        benchmark_id="lpfqa",
        dataset_id="m-a-p/LPFQA",
        split="train",
        metric_id="lpfqa_score",
        requires_judge=True,
        builder=_build_lpfqa_benchmark,
        registrar=_register_lpfqa,
        summarizer=_summarize_lpfqa,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinLPFQADatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
    "hle": BuiltinBenchmarkDefinition(
        benchmark_id="hle",
        dataset_id="cais/hle",
        split="test",
        metric_id="hle_accuracy",
        requires_judge=True,
        builder=_build_hle_benchmark,
        registrar=_register_hle,
        summarizer=_summarize_hle,
        dataset_provider_factory=lambda _definition, huggingface_loader=None: (
            BuiltinHLEDatasetProvider(
                huggingface_loader=huggingface_loader,
            )
        ),
    ),
}


__all__ = [
    "BuiltinBenchmarkDefinition",
    "BuiltinBenchmarkRuntimeConfig",
    "StarterDatasetProvider",
    "StarterNormalizedRows",
    "build_builtin_benchmark_project",
    "build_starter_registry",
    "get_builtin_benchmark",
    "inspect_huggingface_dataset",
    "list_builtin_benchmarks",
    "load_huggingface_rows",
    "load_local_rows",
    "register_starter_engine",
    "register_starter_metrics",
]
