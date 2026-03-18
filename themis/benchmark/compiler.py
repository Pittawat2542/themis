"""Compiler from benchmark-first authoring specs to the private execution IR."""

from __future__ import annotations

from themis.benchmark.specs import BenchmarkSpec, PromptVariantSpec, SliceSpec
from themis.errors import SpecValidationError
from themis.specs.experiment import ExperimentSpec, PromptTemplateSpec
from themis.specs.foundational import (
    EvaluationSpec,
    ExtractorChainSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.types.enums import ErrorCode


def compile_benchmark(benchmark: BenchmarkSpec) -> ExperimentSpec:
    """Lower a benchmark spec into the private experiment/task execution IR."""

    prompt_templates = [
        PromptTemplateSpec(
            id=variant.id,
            family=variant.family,
            messages=variant.messages,
            variables=variant.variables,
        )
        for variant in benchmark.prompt_variants
    ]
    variants_by_id = {variant.id: variant for variant in benchmark.prompt_variants}
    tasks: list[TaskSpec] = []
    for slice_spec in benchmark.slices:
        allowed_prompt_ids = _allowed_prompt_ids(slice_spec, variants_by_id)
        tasks.append(
            TaskSpec(
                task_id=slice_spec.slice_id,
                dataset=slice_spec.dataset,
                dataset_query=slice_spec.dataset_query,
                generation=slice_spec.generation,
                output_transforms=[
                    OutputTransformSpec(
                        name=parse_spec.name,
                        extractor_chain=ExtractorChainSpec(
                            extractors=parse_spec.extractors
                        ),
                    )
                    for parse_spec in slice_spec.parses
                ],
                evaluations=[
                    EvaluationSpec(
                        name=score_spec.name,
                        transform=score_spec.parse,
                        metrics=score_spec.metrics,
                    )
                    for score_spec in slice_spec.scores
                ],
                benchmark_id=benchmark.benchmark_id,
                slice_id=slice_spec.slice_id,
                dimensions=slice_spec.dimensions,
                allowed_prompt_template_ids=allowed_prompt_ids,
                prompt_family_filters=(
                    list(slice_spec.prompt_families)
                    if slice_spec.prompt_families
                    else None
                ),
            )
        )
    return ExperimentSpec(
        models=benchmark.models,
        tasks=tasks,
        prompt_templates=prompt_templates,
        inference_grid=benchmark.inference_grid,
        num_samples=benchmark.num_samples,
    )


def _allowed_prompt_ids(
    slice_spec: SliceSpec,
    variants_by_id: dict[str, PromptVariantSpec],
) -> list[str]:
    if slice_spec.prompt_variant_ids:
        missing_variant_ids = sorted(
            variant_id
            for variant_id in slice_spec.prompt_variant_ids
            if variant_id not in variants_by_id
        )
        if missing_variant_ids:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(
                    f"Slice '{slice_spec.slice_id}' references unknown prompt "
                    f"variant IDs: {', '.join(missing_variant_ids)}."
                ),
            )
        return list(slice_spec.prompt_variant_ids)
    if slice_spec.prompt_families:
        prompt_families_set = set(slice_spec.prompt_families)
        prompt_ids = [
            variant_id
            for variant_id, variant in variants_by_id.items()
            if variant.family in prompt_families_set
        ]
        if not prompt_ids:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(
                    f"Slice '{slice_spec.slice_id}' matched no prompt variants for "
                    f"families: {', '.join(sorted(slice_spec.prompt_families))}."
                ),
            )
        return prompt_ids
    return list(variants_by_id)
