"""Shared catalog dataclasses and benchmark helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
from pathlib import Path

from themis import (
    BenchmarkSpec,
    BenchmarkDefinition,
    BenchmarkDefinitionConfig,
    DatasetQuerySpec,
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
    build_benchmark_definition_project,
)
from themis.contracts.protocols import DatasetProvider
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorRefSpec,
    GenerationSpec,
    JinjaTransform,
    RenameFieldTransform,
)
from themis.types.enums import DatasetSource, PromptRole
from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

from . import datasets as _datasets
from . import runtime as _runtime

_MCQ_CHOICES = list("ABCDEFGHIJ")

type CatalogRow = dict[str, object]


def load_local_rows(path: Path) -> list[dict[str, object]]:
    return _datasets.load_local_rows(path)


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
) -> list[dict[str, object]]:
    return _datasets.load_huggingface_rows(dataset_id, split, revision)


def inspect_huggingface_dataset(
    dataset_id: str,
    *,
    split: str = "test",
    revision: str | None = None,
    metadata_loader=None,
    row_loader=None,
    max_samples: int = 3,
) -> JSONDict:
    return _datasets.inspect_huggingface_dataset(
        dataset_id,
        split=split,
        revision=revision,
        metadata_loader=metadata_loader,
        row_loader=row_loader,
        max_samples=max_samples,
    )


def _catalog_metadata(definition: BenchmarkDefinition) -> JSONDict:
    return validate_json_dict(
        definition.metadata,
        label=f"{definition.family} benchmark metadata",
    )


def _catalog_metadata_str(definition: BenchmarkDefinition, key: str) -> str:
    value = _catalog_metadata(definition).get(key)
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        f"Built-in benchmark '{definition.benchmark_id}' metadata must define '{key}'."
    )


def _primary_metric_id(definition: BenchmarkDefinition) -> str:
    if definition.primary_metric_id:
        return definition.primary_metric_id
    raise ValueError(
        f"Built-in benchmark '{definition.benchmark_id}' does not define a primary metric."
    )


def build_catalog_benchmark_project(
    *,
    benchmark_id: str,
    model_id: str,
    provider: str,
    storage_root: Path,
    get_catalog_benchmark: Callable[[str], BenchmarkDefinition],
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
    DatasetProvider,
    BenchmarkDefinition,
]:
    definition = get_catalog_benchmark(benchmark_id)
    return build_benchmark_definition_project(
        definition=definition,
        model_id=model_id,
        provider=provider,
        storage_root=storage_root,
        build_registry=_runtime.build_catalog_registry,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        subset=subset,
        dataset_revision=dataset_revision,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
        huggingface_loader=load_huggingface_rows
        if huggingface_loader is None
        else huggingface_loader,
    )


def iter_score_rows(result, metric_id: str) -> list[ScoreRow]:
    return list(
        result.projection_repo.iter_candidate_scores(
            trial_hashes=result.trial_hashes,
            metric_id=metric_id,
            evaluation_hash=getattr(result, "active_evaluation_hash", None),
        )
    )


def mean_summary(metric_id: str, result) -> JSONDict:
    rows = iter_score_rows(result, metric_id)
    count = len(rows)
    mean = sum(row.score for row in rows) / count if count else 0.0
    return _json_dict(
        {"metric_id": metric_id, "count": count, "mean": mean},
        label=f"{metric_id} summary",
    )


def summarize_simpleqa(_definition, result) -> JSONDict:
    rows = iter_score_rows(result, "simpleqa_verified_score")
    count = len(rows)
    if count == 0:
        return _json_dict(
            {
                "metric_id": "simpleqa_verified_score",
                "count": 0,
                "correct_rate": 0.0,
                "attempted_rate": 0.0,
                "accuracy_given_attempted": 0.0,
                "f1": 0.0,
            },
            label="simpleqa summary",
        )
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
    return _json_dict(
        {
            "metric_id": "simpleqa_verified_score",
            "count": count,
            "correct_rate": correct_rate,
            "attempted_rate": attempted_rate,
            "accuracy_given_attempted": accuracy_given_attempted,
            "f1": f1,
        },
        label="simpleqa summary",
    )


def summarize_healthbench(_definition, result) -> JSONDict:
    rows = iter_score_rows(result, "healthbench_score")
    count = len(rows)
    mean_score = sum(row.score for row in rows) / count if count else 0.0
    tag_values: dict[str, list[float]] = {}
    for row in rows:
        for tag in _detail_str_list(row.details.get("example_tags")):
            if isinstance(tag, str):
                tag_values.setdefault(tag, []).append(row.score)
    return _json_dict(
        {
            "metric_id": "healthbench_score",
            "count": count,
            "mean_overall_score": mean_score,
            "tag_means": {
                tag: sum(values) / len(values)
                for tag, values in sorted(tag_values.items())
            },
        },
        label="healthbench summary",
    )


def summarize_hle(_definition, result) -> JSONDict:
    rows = iter_score_rows(result, "hle_accuracy")
    count = len(rows)
    accuracy = sum(row.score for row in rows) / count if count else 0.0
    confidence_interval_half_width = (
        1.96 * math.sqrt(accuracy * (1 - accuracy) / count) if count else 0.0
    )
    confidences = [
        max(
            0.0,
            min(1.0, _coerce_score_float(row.details.get("confidence"), 100.0) / 100.0),
        )
        for row in rows
    ]
    truths = [float(bool(row.details.get("correct", False))) for row in rows]
    calibration_error = hle_calibration_error(
        confidences=confidences,
        truths=truths,
        accuracy=accuracy,
    )
    scan_stats = _detail_mapping(getattr(result, "_builtin_scan_stats", {}) or {})
    return _json_dict(
        {
            "metric_id": "hle_accuracy",
            "count": count,
            "accuracy": accuracy,
            "confidence_interval_half_width": confidence_interval_half_width,
            "calibration_error": calibration_error,
            "skipped_image_count": _coerce_score_int(
                scan_stats.get("skipped_image_count"),
                0,
            ),
        },
        label="hle summary",
    )


def hle_calibration_error(
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


def summarize_mcq(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_math(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def summarize_lpfqa(definition, result) -> JSONDict:
    return mean_summary(_primary_metric_id(definition), result)


def make_dataset_query(config: BenchmarkDefinitionConfig) -> DatasetQuerySpec:
    if config.subset is None:
        return DatasetQuerySpec()
    return DatasetQuerySpec.subset(config.subset, seed=config.seed)


def mcq_dataset_spec(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
    *,
    expected_source_field: str,
) -> DatasetSpec:
    return DatasetSpec(
        source=DatasetSource.HUGGINGFACE,
        dataset_id=_catalog_metadata_str(definition, "dataset_id"),
        split=_catalog_metadata_str(definition, "split"),
        revision=config.dataset_revision,
        transforms=[
            RenameFieldTransform(
                field="expected",
                source_field=expected_source_field,
            ),
            JinjaTransform(
                field="prompt_text",
                template=(
                    "Question:\n{question}\n\nOptions:\n{options_text}\n\n"
                    "Return the best option letter only."
                ),
            ),
        ],
    )


def math_dataset_spec(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> DatasetSpec:
    return DatasetSpec(
        source=DatasetSource.HUGGINGFACE,
        dataset_id=_catalog_metadata_str(definition, "dataset_id"),
        split=_catalog_metadata_str(definition, "split"),
        revision=config.dataset_revision,
        transforms=[
            JinjaTransform(
                field="prompt_text",
                template=(
                    "Solve the following math problem. "
                    "Return only the final answer in \\boxed{{...}}.\n\n"
                    "Problem:\n{problem}"
                ),
            )
        ],
    )


def build_mcq_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
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
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=mcq_dataset_spec(
                    definition,
                    config,
                    expected_source_field=expected_source_field,
                ),
                dataset_query=make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                parses=[
                    ParseSpec(
                        name="parsed",
                        extractors=[
                            ExtractorRefSpec(
                                id="choice_letter",
                                config=_json_dict(
                                    {"choices": _MCQ_CHOICES},
                                    label="mcq extractor config",
                                ),
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


def build_math_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=math_dataset_spec(definition, config),
                dataset_query=make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                parses=[
                    ParseSpec(
                        name="parsed",
                        extractors=[ExtractorRefSpec(id="math_answer")],
                    )
                ],
                scores=[
                    ScoreSpec(
                        name="default",
                        parse="parsed",
                        metrics=["math_equivalence"],
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


def build_simpleqa_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                    transforms=[
                        RenameFieldTransform(
                            field="prompt_text",
                            source_field="problem",
                        )
                    ],
                ),
                dataset_query=make_dataset_query(config),
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


def build_healthbench_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                ),
                dataset_query=make_dataset_query(config),
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


def build_lpfqa_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                    transforms=[
                        RenameFieldTransform(
                            field="prompt_text",
                            source_field="prompt",
                        )
                    ],
                ),
                dataset_query=make_dataset_query(config),
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


def build_hle_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_runtime._provider_model_extras(config.provider),
            )
        ],
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                    transforms=[
                        RenameFieldTransform(
                            field="expected",
                            source_field="answer",
                        ),
                        JinjaTransform(
                            field="prompt_text",
                            template=(
                                "Your response should be in the following format:\n"
                                "Explanation: your explanation for your answer choice\n"
                                "Answer: your chosen answer\n"
                                "Confidence: your confidence score between 0% and 100% "
                                "for your answer\n\nQuestion:\n{question}"
                            ),
                        ),
                    ],
                ),
                dataset_query=make_dataset_query(config),
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


def register_mcq(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    if not registry.has_metric("choice_accuracy"):
        registry.register_metric("choice_accuracy", _runtime.ChoiceAccuracyMetric)


def register_math(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    if not registry.has_metric("math_equivalence"):
        registry.register_metric("math_equivalence", _runtime.MathEquivalenceMetric)


def register_simpleqa(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from .benchmarks.simpleqa_verified.metric import SimpleQAVerifiedJudgeMetric

    registry.register_metric(
        "simpleqa_verified_score",
        partial(
            SimpleQAVerifiedJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_healthbench(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from .benchmarks.healthbench.metric import HealthBenchRubricMetric

    registry.register_metric(
        "healthbench_score",
        partial(
            HealthBenchRubricMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_lpfqa(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from .benchmarks.lpfqa.metric import LPFQAJudgeMetric

    registry.register_metric(
        "lpfqa_score",
        partial(
            LPFQAJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def register_hle(
    _definition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    from .benchmarks.hle.metric import HLEJudgeMetric

    registry.register_metric(
        "hle_accuracy",
        partial(
            HLEJudgeMetric,
            judge_model_id=str(config.judge_model_id),
            judge_provider=str(config.judge_provider),
        ),
    )


def render_healthbench_preview(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
    sample: dict[str, object],
) -> list[JSONDict]:
    del definition, config
    return [
        _json_dict(
            {
                "prompt_variant_id": "healthbench-default",
                "messages": _datasets._prompt_messages_from_context(sample),
                "follow_up_turns": [],
            },
            label="healthbench preview",
        )
    ]


def _json_dict(value: object, *, label: str) -> JSONDict:
    return validate_json_dict(value, label=label)


def _detail_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _detail_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _coerce_score_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_score_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
