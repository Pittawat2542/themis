from __future__ import annotations

import pytest

from pydantic import ValidationError

from themis import (
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
from themis.benchmark.specs import DatasetSliceSpec
from themis.benchmark.compiler import compile_benchmark
from themis.errors import SpecValidationError
from themis.orchestration.trial_planner import TrialPlanner
from themis.specs.experiment import DataItemContext
from themis.specs.experiment import ExperimentSpec
from themis.specs.experiment import PromptTemplateSpec
from themis.types.enums import DatasetSource, PromptRole, SamplingKind
from themis.specs.foundational import DatasetSpec, ExtractorRefSpec, GenerationSpec


class RecordingDatasetProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[DatasetSliceSpec, DatasetQuerySpec]] = []

    def scan(self, slice_spec, query):
        self.calls.append((slice_spec, query))
        return [
            DataItemContext(
                item_id="item-1",
                payload={"question": "2 + 2", "answer": "4"},
                metadata={"difficulty": "easy"},
            )
        ]


def test_compile_benchmark_maps_prompt_applicability_and_slice_metadata() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="math-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec(count=5, seed=7),
                dimensions={"domain": "math", "format": "qa"},
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                parses=[
                    ParseSpec(
                        name="parsed",
                        extractors=[ExtractorRefSpec(id="first_number")],
                    )
                ],
                scores=[
                    ScoreSpec(name="default", parse="parsed", metrics=["exact_match"])
                ],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            ),
            PromptVariantSpec(
                id="mcq-default",
                family="mcq",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(benchmark)
    task = experiment.tasks[0]

    assert task.benchmark_id == "math-bench"
    assert task.slice_id == "arithmetic"
    assert task.dimensions == {"domain": "math", "format": "qa"}
    assert task.allowed_prompt_template_ids == ["qa-default"]
    assert isinstance(task.dataset_query, DatasetQuerySpec)
    assert task.dataset_query.count == 5


def test_trial_planner_uses_dataset_provider_query_pushdown_and_prompt_filters() -> (
    None
):
    benchmark = BenchmarkSpec(
        benchmark_id="math-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec(count=3, seed=11),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Question: {item.question}"
                    )
                ],
            ),
            PromptVariantSpec(
                id="ignored-variant",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Ignored: {item.question}"
                    )
                ],
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    provider = RecordingDatasetProvider()
    planner = TrialPlanner(dataset_provider=provider)

    planned_trials = planner.plan_benchmark(benchmark)

    assert len(planned_trials) == 1
    assert planned_trials[0].trial_spec.prompt.id == "qa-default"
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == DatasetSliceSpec(
        benchmark_id="math-bench",
        slice_id="arithmetic",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        dimensions={},
    )
    assert provider.calls[0][1] == DatasetQuerySpec(count=3, seed=11)


def test_trial_planner_rejects_unmatched_prompt_family_filters() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="math-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_families=["missing-family"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Question: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    planner = TrialPlanner(dataset_provider=RecordingDatasetProvider())

    with pytest.raises(SpecValidationError, match="missing-family"):
        planner.plan_benchmark(benchmark)


def test_trial_planner_rejects_unmatched_prompt_variant_ids() -> None:
    with pytest.raises(ValidationError, match="missing-variant"):
        BenchmarkSpec(
            benchmark_id="math-bench",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="arithmetic",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    prompt_variant_ids=["missing-variant"],
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    family="qa",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER, content="Question: {item.question}"
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=32)]
            ),
        )


def test_slice_spec_rejects_duplicate_parse_names() -> None:
    with pytest.raises(ValidationError, match="duplicate parse name"):
        SliceSpec(
            slice_id="arithmetic",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            parses=[
                ParseSpec(
                    name="parsed",
                    extractors=[ExtractorRefSpec(id="first_number")],
                ),
                ParseSpec(
                    name="parsed",
                    extractors=[ExtractorRefSpec(id="second_number")],
                ),
            ],
        )


def test_slice_spec_rejects_duplicate_score_names() -> None:
    with pytest.raises(ValidationError, match="duplicate score name"):
        SliceSpec(
            slice_id="arithmetic",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            scores=[
                ScoreSpec(name="default", metrics=["exact_match"]),
                ScoreSpec(name="default", metrics=["accuracy"]),
            ],
        )


def test_slice_spec_rejects_scores_that_reference_unknown_parse_names() -> None:
    with pytest.raises(ValidationError, match="unknown parse"):
        SliceSpec(
            slice_id="arithmetic",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            parses=[
                ParseSpec(
                    name="parsed",
                    extractors=[ExtractorRefSpec(id="first_number")],
                )
            ],
            scores=[ScoreSpec(name="default", parse="missing", metrics=["accuracy"])],
        )


def test_dataset_query_spec_rejects_item_ids_with_count_based_sampling() -> None:
    with pytest.raises(ValidationError, match="item_ids"):
        DatasetQuerySpec(
            kind=SamplingKind.SUBSET,
            count=2,
            item_ids=["item-1"],
        )

    with pytest.raises(ValidationError, match="item_ids"):
        DatasetQuerySpec(
            kind=SamplingKind.STRATIFIED,
            count=2,
            strata_field="difficulty",
            item_ids=["item-1"],
        )


def test_trial_planner_resolves_prompt_selectors_before_dataset_access() -> None:
    provider = RecordingDatasetProvider()
    planner = TrialPlanner(dataset_provider=provider)
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            compile_benchmark(
                BenchmarkSpec(
                    benchmark_id="math-bench",
                    models=[ModelSpec(model_id="demo-model", provider="demo")],
                    slices=[
                        SliceSpec(
                            slice_id="arithmetic",
                            dataset=DatasetSpec(source=DatasetSource.MEMORY),
                            generation=GenerationSpec(),
                            scores=[
                                ScoreSpec(
                                    name="default",
                                    metrics=["exact_match"],
                                )
                            ],
                        )
                    ],
                    prompt_variants=[
                        PromptVariantSpec(
                            id="qa-default",
                            family="qa",
                            messages=[
                                PromptMessage(
                                    role=PromptRole.USER,
                                    content="Question: {item.question}",
                                )
                            ],
                        )
                    ],
                    inference_grid=InferenceGridSpec(
                        params=[InferenceParamsSpec(max_tokens=32)]
                    ),
                )
            )
            .tasks[0]
            .model_copy(update={"allowed_prompt_template_ids": ["missing-variant"]})
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    with pytest.raises(SpecValidationError, match="missing-variant"):
        planner.plan_experiment(experiment)

    assert provider.calls == []
