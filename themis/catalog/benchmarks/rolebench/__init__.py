"""Catalog definition for RoleBench."""

from __future__ import annotations

from dataclasses import replace

from themis import (
    BenchmarkDefinition,
    BenchmarkDefinitionConfig,
    BenchmarkSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    PluginRegistry,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import DatasetSource, PromptRole
from themis.types.events import ScoreRow
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

from ...common.builders import make_dataset_query
from ...common.summaries import iter_score_rows
from ...runtime._provider import _provider_model_extras
from .dataset import BuiltinRoleBenchDatasetProvider
from .metric import RoleBenchRougeMetric

_ROLEBENCH_SYSTEM_PROMPT = (
    "You are {item.role}, your description is: {item.desc}. "
    "Now please answer some questions to accurately show your personality traits! "
    "Your speaking style should fully imitate the personality role assigned to you! "
    "Please do not expose that you are an artificial intelligence model or a "
    "language model, you must always remember that you are only assigned one "
    "personality role. Don't be verbose or too formal or polite when speaking."
)
_SUPPORTED_ROLEBENCH_VARIANT_IDS = (
    "instruction_generalization_eng",
    "role_generalization_eng",
)
_VARIANT_PATHS = {
    "instruction_generalization_eng": "instruction-generalization",
    "role_generalization_eng": "role-generalization",
}


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "role": "Wizard",
            "desc": "Speaks cryptically.",
            "question": "What would you say to a lost traveler?",
            "generated": ["Follow the silver river until dawn."],
            "subset": "general",
            "source_line_number": 1,
        }
    ]


def _build_rolebench_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    variant_ids = _rolebench_variant_ids(definition)
    slices: list[SliceSpec] = []
    prompt_variants: list[PromptVariantSpec] = []
    for variant_id in variant_ids:
        slice_id = f"rolebench-{variant_id}"
        prompt_variant_id = f"{slice_id}-default"
        slices.append(
            SliceSpec(
                slice_id=slice_id,
                dimensions={"rolebench_variant": variant_id},
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=str(definition.metadata["dataset_id"]),
                    config_name=variant_id,
                    split=str(definition.metadata["split"]),
                    revision=config.dataset_revision,
                ),
                dataset_query=make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[
                    ScoreSpec(
                        name="default",
                        metrics=["rolebench_rouge_l_f1"],
                    )
                ],
            )
        )
        prompt_variants.append(
            PromptVariantSpec(
                id=prompt_variant_id,
                family=definition.benchmark_id,
                messages=[
                    PromptMessage(
                        role=PromptRole.SYSTEM,
                        content=_ROLEBENCH_SYSTEM_PROMPT,
                    ),
                    PromptMessage(
                        role=PromptRole.USER,
                        content="{item.question}",
                    ),
                ],
            )
        )
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=[
            ModelSpec(
                model_id=config.model_id,
                provider=config.provider,
                extras=_provider_model_extras(config.provider),
            )
        ],
        slices=slices,
        prompt_variants=prompt_variants,
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


def _register_rolebench(
    _definition: BenchmarkDefinition,
    registry: PluginRegistry,
    config: BenchmarkDefinitionConfig,
) -> None:
    del config
    if not registry.has_metric("rolebench_rouge_l_f1"):
        registry.register_metric("rolebench_rouge_l_f1", RoleBenchRougeMetric)


def _summarize_rolebench(definition: BenchmarkDefinition, result) -> JSONDict:
    metric_id = "rolebench_rouge_l_f1"
    rows = iter_score_rows(result, metric_id)
    count = len(rows)
    mean = sum(row.score for row in rows) / count if count else 0.0
    variant_ids = _rolebench_variant_ids(definition)
    if len(variant_ids) == 1:
        return _json_dict(
            {
                "metric_id": metric_id,
                "count": count,
                "mean": mean,
            },
            label="rolebench summary",
        )
    summaries_by_trial_hash = {}
    if hasattr(result, "iter_trial_summaries"):
        summaries_by_trial_hash = {
            row.trial_hash: row
            for row in result.iter_trial_summaries()  # type: ignore[attr-defined]
        }
    grouped_rows: dict[str, list[ScoreRow]] = {
        variant_id: [] for variant_id in variant_ids
    }
    for row in rows:
        variant_id = _resolve_rolebench_variant_id(
            row,
            summaries_by_trial_hash.get(row.trial_hash),
        )
        if variant_id in grouped_rows:
            grouped_rows[variant_id].append(row)
    return _json_dict(
        {
            "metric_id": metric_id,
            "count": count,
            "mean": mean,
            "variant_ids": list(variant_ids),
            "variants": {
                variant_id: _json_dict(
                    {
                        "metric_id": metric_id,
                        "count": len(grouped_rows[variant_id]),
                        "mean": (
                            sum(row.score for row in grouped_rows[variant_id])
                            / len(grouped_rows[variant_id])
                            if grouped_rows[variant_id]
                            else 0.0
                        ),
                    },
                    label=f"rolebench {variant_id} summary",
                )
                for variant_id in variant_ids
            },
        },
        label="rolebench summary",
    )


def _resolve_rolebench_variant_id(
    row: object, summary_row: object | None
) -> str | None:
    if summary_row is not None:
        dimensions = getattr(summary_row, "dimensions", {}) or {}
        if isinstance(dimensions, dict):
            candidate = dimensions.get("rolebench_variant")
            if isinstance(candidate, str):
                return candidate
        slice_id = getattr(summary_row, "slice_id", None)
        if isinstance(slice_id, str) and slice_id.startswith("rolebench-"):
            return slice_id.removeprefix("rolebench-")
    prompt_variant_id = getattr(row, "prompt_variant_id", None)
    if isinstance(prompt_variant_id, str) and prompt_variant_id.startswith(
        "rolebench-"
    ):
        return prompt_variant_id.removeprefix("rolebench-").removesuffix("-default")
    return None


def _rolebench_variant_ids(definition: BenchmarkDefinition) -> list[str]:
    raw_value = definition.metadata.get(
        "variant_ids", list(_SUPPORTED_ROLEBENCH_VARIANT_IDS)
    )
    if not isinstance(raw_value, list) or not all(
        isinstance(item, str) and item for item in raw_value
    ):
        raise ValueError(
            f"Built-in benchmark '{definition.benchmark_id}' metadata must define non-empty RoleBench variant ids."
        )
    return [str(item) for item in raw_value]


def supported_rolebench_variant_ids() -> tuple[str, ...]:
    return _SUPPORTED_ROLEBENCH_VARIANT_IDS


DEFINITION = BenchmarkDefinition(
    benchmark_id="rolebench",
    family="catalog",
    primary_metric_id="rolebench_rouge_l_f1",
    requires_judge=False,
    metadata={
        "dataset_id": "ZenMoore/RoleBench",
        "split": "test",
        "variant_ids": list(_SUPPORTED_ROLEBENCH_VARIANT_IDS),
        "variant_paths": dict(_VARIANT_PATHS),
    },
    builder=_build_rolebench_benchmark,
    registrar=_register_rolebench,
    summarizer=_summarize_rolebench,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinRoleBenchDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)


def build_rolebench_definition(variant_ids: list[str]) -> BenchmarkDefinition:
    benchmark_id = "rolebench"
    if len(variant_ids) == 1:
        benchmark_id = f"rolebench:{variant_ids[0]}"
    return replace(
        DEFINITION,
        benchmark_id=benchmark_id,
        metadata={
            **DEFINITION.metadata,
            "variant_ids": list(variant_ids),
        },
    )


def _json_dict(value: object, *, label: str) -> JSONDict:
    return validate_json_dict(value, label=label)
