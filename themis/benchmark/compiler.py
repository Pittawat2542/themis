"""Compiler from benchmark-first authoring specs to the private execution IR."""

from __future__ import annotations

from collections.abc import Sequence

from themis.benchmark.specs import BenchmarkSpec, PromptVariantSpec, SliceSpec
from themis.errors import SpecValidationError
from themis.specs.experiment import ExperimentSpec, PromptTemplateSpec
from themis.specs.foundational import (
    EvaluationSpec,
    ExtractorChainSpec,
    OutputTransformSpec,
    TaskSpec,
    ToolSpec,
)
from themis.types.enums import ErrorCode


def merge_tool_specs(
    base_tools: Sequence[ToolSpec],
    override_tools: Sequence[ToolSpec],
) -> list[ToolSpec]:
    """Merge ordered tool declarations with same-id overrides."""

    merged: dict[str, ToolSpec] = {tool.id: tool for tool in base_tools}
    for tool in override_tools:
        merged[tool.id] = tool
    ordered_ids = [tool.id for tool in base_tools]
    for tool in override_tools:
        if tool.id not in ordered_ids:
            ordered_ids.append(tool.id)
    return [merged[tool_id] for tool_id in ordered_ids]


def normalize_benchmark_spec(
    benchmark: BenchmarkSpec,
    *,
    project_tools: Sequence[ToolSpec] = (),
) -> BenchmarkSpec:
    """Return a benchmark with project and benchmark tool declarations merged."""

    return benchmark.model_copy(
        update={
            "tools": merge_tool_specs(project_tools, benchmark.tools),
        }
    )


def compile_benchmark(
    benchmark: BenchmarkSpec,
    *,
    project_tools: Sequence[ToolSpec] = (),
) -> ExperimentSpec:
    """Lower a benchmark spec into the private experiment/task execution IR."""

    benchmark = normalize_benchmark_spec(benchmark, project_tools=project_tools)
    prompt_templates = [
        PromptTemplateSpec(
            id=variant.id,
            family=variant.family,
            messages=variant.messages,
            follow_up_turns=variant.follow_up_turns,
            variables=variant.variables,
        )
        for variant in benchmark.prompt_variants
    ]
    variants_by_id = {variant.id: variant for variant in benchmark.prompt_variants}
    tool_ids = {tool.id for tool in benchmark.tools}
    tasks: list[TaskSpec] = []
    for slice_spec in benchmark.slices:
        allowed_prompt_ids = _allowed_prompt_ids(slice_spec, variants_by_id)
        selected_tool_ids = _validated_tool_ids(
            benchmark_id=benchmark.benchmark_id,
            slice_spec=slice_spec,
            known_tool_ids=tool_ids,
        )
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
                tool_ids=selected_tool_ids,
            )
        )
    return ExperimentSpec(
        models=benchmark.models,
        tasks=tasks,
        prompt_templates=prompt_templates,
        tools=benchmark.tools,
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


def _validated_tool_ids(
    *,
    benchmark_id: str,
    slice_spec: SliceSpec,
    known_tool_ids: set[str],
) -> list[str]:
    unknown_tool_ids = sorted(set(slice_spec.tool_ids) - known_tool_ids)
    if unknown_tool_ids:
        raise ValueError(
            f"BenchmarkSpec '{benchmark_id}' slice '{slice_spec.slice_id}' "
            f"references unknown tool id(s): {', '.join(unknown_tool_ids)}."
        )
    return list(slice_spec.tool_ids)
