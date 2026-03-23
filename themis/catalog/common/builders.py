"""Catalog benchmark builders and dataset spec helpers."""

from __future__ import annotations

from themis import (
    BenchmarkDefinition,
    BenchmarkDefinitionConfig,
    BenchmarkSpec,
    DatasetQuerySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    ParseSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorRefSpec,
    GenerationSpec,
    JinjaTransform,
    RenameFieldTransform,
    TransformSpec,
)
from themis.types.enums import DatasetSource, PromptRole
from themis.types.json_types import JSONDict

from ..runtime._provider import _provider_model_extras
from ._shared import (
    _catalog_metadata,
    _catalog_metadata_optional_str,
    _catalog_metadata_str,
    _hle_prompt_template,
    _hle_variant_ids,
    _json_dict,
    _primary_metric_id,
)

_MCQ_CHOICES = list("ABCDEFGHIJ")


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
        config_name=_catalog_metadata_optional_str(definition, "config_name"),
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
        config_name=_catalog_metadata_optional_str(definition, "config_name"),
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
        models=_default_model_specs(config),
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
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_math_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
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
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_simpleqa_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
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
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_healthbench_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
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
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_lpfqa_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
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
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_hle_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    variant_ids = _hle_variant_ids(definition)
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=f"hle-{variant_id}",
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                    transforms=[
                        RenameFieldTransform(
                            field="expected",
                            source_field="answer",
                        ),
                        JinjaTransform(
                            field="prompt_text",
                            template=_hle_prompt_template(variant_id),
                        ),
                    ],
                ),
                dataset_query=make_dataset_query(config),
                dimensions={"hle_variant": variant_id},
                prompt_variant_ids=[f"hle-{variant_id}-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["hle_accuracy"])],
            )
            for variant_id in variant_ids
        ],
        prompt_variants=[
            _single_prompt_variant(
                f"hle-{variant_id}-default",
                family="hle",
                variables={"hle_variant": variant_id},
            )
            for variant_id in variant_ids
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_frontierscience_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                ),
                dataset_query=make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="judge", metrics=["frontierscience_score"])],
            )
        ],
        prompt_variants=[
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_procbench_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    metadata = _catalog_metadata(definition)
    raw_task_ids = metadata.get("task_ids", [])
    task_ids = (
        [str(item) for item in raw_task_ids] if isinstance(raw_task_ids, list) else []
    )
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=f"procbench-{task_id}",
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=task_id,
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                ),
                dataset_query=make_dataset_query(config),
                dimensions={"task_name": task_id},
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[
                    ScoreSpec(name="default", metrics=["procbench_final_accuracy"])
                ],
            )
            for task_id in task_ids
        ],
        prompt_variants=[
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
                content=(
                    "{item.prompt_text}\n\n"
                    "Return only the final answer. "
                    "If the answer is a list, return a JSON list."
                ),
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def build_codeforces_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    return _build_code_generation_benchmark(
        definition,
        config,
        prompt_source_field="prompt",
        metric_id="codeforces_pass_rate",
    )


def build_aethercode_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    return _build_code_generation_benchmark(
        definition,
        config,
        prompt_source_field=None,
        metric_id="aethercode_pass_rate",
    )


def build_livecodebench_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    return _build_code_generation_benchmark(
        definition,
        config,
        prompt_source_field=None,
        metric_id="livecodebench_pass_rate",
    )


def build_humaneval_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
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
                dimensions={
                    "humaneval_variant": str(
                        _catalog_metadata(definition).get("variant", "base")
                    )
                },
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[
                    ScoreSpec(
                        name="execution",
                        metrics=[_primary_metric_id(definition)],
                    )
                ],
            )
        ],
        prompt_variants=[
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def _build_code_generation_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
    *,
    prompt_source_field: str | None,
    metric_id: str,
) -> BenchmarkSpec:
    prompt_variant_id = f"{definition.benchmark_id}-default"
    transforms: list[TransformSpec] = []
    if prompt_source_field is not None:
        transforms.append(
            RenameFieldTransform(
                field="prompt_text",
                source_field=prompt_source_field,
            )
        )
    return BenchmarkSpec(
        benchmark_id=definition.benchmark_id,
        models=_default_model_specs(config),
        slices=[
            SliceSpec(
                slice_id=definition.benchmark_id,
                dataset=DatasetSpec(
                    source=DatasetSource.HUGGINGFACE,
                    dataset_id=_catalog_metadata_str(definition, "dataset_id"),
                    config_name=_catalog_metadata_optional_str(
                        definition, "config_name"
                    ),
                    split=_catalog_metadata_str(definition, "split"),
                    revision=config.dataset_revision,
                    transforms=transforms,
                ),
                dataset_query=make_dataset_query(config),
                prompt_variant_ids=[prompt_variant_id],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="execution", metrics=[metric_id])],
            )
        ],
        prompt_variants=[
            _single_prompt_variant(
                prompt_variant_id,
                family=definition.benchmark_id,
            )
        ],
        inference_grid=_default_inference_grid(config),
    )


def _default_model_specs(
    config: BenchmarkDefinitionConfig,
) -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id=config.model_id,
            provider=config.provider,
            extras=_provider_model_extras(config.provider),
        )
    ]


def _default_inference_grid(
    config: BenchmarkDefinitionConfig,
) -> InferenceGridSpec:
    return InferenceGridSpec(
        params=[
            InferenceParamsSpec(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
            )
        ]
    )


def _single_prompt_variant(
    prompt_variant_id: str,
    *,
    family: str,
    content: str = "{item.prompt_text}",
    role: PromptRole = PromptRole.USER,
    variables: JSONDict | None = None,
) -> PromptVariantSpec:
    if variables is None:
        return PromptVariantSpec(
            id=prompt_variant_id,
            family=family,
            messages=[PromptMessage(role=role, content=content)],
        )
    return PromptVariantSpec(
        id=prompt_variant_id,
        family=family,
        variables=variables,
        messages=[PromptMessage(role=role, content=content)],
    )
