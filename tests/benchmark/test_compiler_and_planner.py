from __future__ import annotations

import pytest

from pydantic import ValidationError

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    McpServerSpec,
    ModelSpec,
    ParseSpec,
    PromptMessage,
    PromptTurnSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    ToolSpec,
)
from themis.benchmark.specs import DatasetSliceSpec
from themis.benchmark.compiler import compile_benchmark
from themis.errors import SpecValidationError
from themis.orchestration.trial_planner import TrialPlanner
from themis.specs.experiment import DataItemContext
from themis.specs.experiment import ExperimentSpec
from themis.specs.experiment import PromptTemplateSpec
from themis.types.enums import DatasetSource, PromptRole, SamplingKind
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorRefSpec,
    GenerationSpec,
    TaskSpec,
)


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


def test_compile_benchmark_preserves_follow_up_turns() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.DEVELOPER,
                        content="Use tools carefully.",
                    ),
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Solve: {item.question}",
                    ),
                ],
                follow_up_turns=[
                    PromptTurnSpec(
                        messages=[
                            PromptMessage(
                                role=PromptRole.USER,
                                content="Continue with {runtime.run_labels[phase]}",
                            )
                        ]
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(benchmark)

    assert experiment.prompt_templates == [
        PromptTemplateSpec(
            id="agent-default",
            family="agent",
            messages=[
                PromptMessage(
                    role=PromptRole.DEVELOPER,
                    content="Use tools carefully.",
                ),
                PromptMessage(
                    role=PromptRole.USER,
                    content="Solve: {item.question}",
                ),
            ],
            follow_up_turns=[
                PromptTurnSpec(
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Continue with {runtime.run_labels[phase]}",
                        )
                    ]
                )
            ],
        )
    ]


def test_compile_benchmark_preserves_tool_ids() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                tool_ids=["search", "calculator"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            ),
            ToolSpec(
                id="calculator",
                description="Calculate",
                input_schema={"type": "object"},
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(benchmark)

    assert [tool.id for tool in experiment.tools] == ["search", "calculator"]
    assert experiment.tasks[0].tool_ids == ["search", "calculator"]


def test_compile_benchmark_preserves_mcp_server_ids() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                mcp_server_ids=["dice"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
                require_approval="never",
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(benchmark)

    assert [server.id for server in experiment.mcp_servers] == ["dice"]
    assert experiment.tasks[0].mcp_server_ids == ["dice"]


def test_compile_benchmark_rejects_unknown_slice_tool_ids() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                tool_ids=["missing-tool"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    with pytest.raises(ValueError, match="agent-bench.*agentic.*missing-tool"):
        compile_benchmark(benchmark)


def test_compile_benchmark_rejects_unknown_slice_mcp_server_ids() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                mcp_server_ids=["missing-mcp"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    with pytest.raises(ValueError, match="agent-bench.*agentic.*missing-mcp"):
        compile_benchmark(benchmark)


def test_compile_benchmark_merges_project_tools_with_benchmark_overrides() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                tool_ids=["search", "calculator", "lookup"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Benchmark search override",
                input_schema={"type": "object"},
            ),
            ToolSpec(
                id="lookup",
                description="Benchmark lookup",
                input_schema={"type": "object"},
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(
        benchmark,
        project_tools=[
            ToolSpec(
                id="search",
                description="Project search",
                input_schema={"type": "object"},
            ),
            ToolSpec(
                id="calculator",
                description="Project calculator",
                input_schema={"type": "object"},
            ),
        ],
    )

    assert [tool.id for tool in experiment.tools] == ["search", "calculator", "lookup"]
    assert {tool.id: tool.description for tool in experiment.tools} == {
        "search": "Benchmark search override",
        "calculator": "Project calculator",
        "lookup": "Benchmark lookup",
    }


def test_compile_benchmark_merges_project_mcp_servers_with_benchmark_overrides() -> (
    None
):
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                mcp_server_ids=["dice", "calendar", "docs"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice-override",
                server_url="https://example.com/dice",
                require_approval="never",
            ),
            McpServerSpec(
                id="docs",
                server_label="docs",
                server_url="https://example.com/docs",
                require_approval="never",
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    experiment = compile_benchmark(
        benchmark,
        project_tools=[],
        project_mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_url="https://dmcp-server.deno.dev/sse",
                require_approval="never",
            ),
            McpServerSpec(
                id="calendar",
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
                require_approval="never",
            ),
        ],
    )

    assert [server.id for server in experiment.mcp_servers] == [
        "dice",
        "calendar",
        "docs",
    ]
    assert {server.id: server.server_label for server in experiment.mcp_servers} == {
        "dice": "dice-override",
        "calendar": "google_calendar",
        "docs": "docs",
    }


def test_compile_benchmark_rejects_unknown_slice_tool_ids_from_unvalidated_copy() -> (
    None
):
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                tool_ids=["search"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    ).model_copy(
        update={
            "slices": [
                SliceSpec(
                    slice_id="agentic",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    prompt_variant_ids=["agent-default"],
                    generation=GenerationSpec(),
                    tool_ids=["search"],
                ).model_copy(update={"tool_ids": ["missing-tool"]})
            ]
        }
    )

    with pytest.raises(ValueError, match="agent-bench.*agentic.*missing-tool"):
        compile_benchmark(benchmark)


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


def test_trial_planner_materializes_selected_tools_on_trial_specs() -> None:
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                tool_ids=["search"],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="agent-default",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            ),
            ToolSpec(
                id="calculator",
                description="Calculate",
                input_schema={"type": "object"},
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    planner = TrialPlanner(dataset_provider=RecordingDatasetProvider())

    planned_trials = planner.plan_experiment(experiment)

    assert len(planned_trials) == 1
    assert [tool.id for tool in planned_trials[0].trial_spec.tools] == ["search"]


def test_trial_planner_rejects_unknown_task_tool_ids() -> None:
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                tool_ids=["missing-tool"],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="agent-default",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    planner = TrialPlanner(dataset_provider=RecordingDatasetProvider())

    with pytest.raises(SpecValidationError, match="missing-tool"):
        planner.plan_experiment(experiment)


def test_trial_planner_plan_benchmark_merges_project_tools() -> None:
    benchmark = BenchmarkSpec(
        benchmark_id="agent-bench",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="agentic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                tool_ids=["search", "calculator"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        tools=[
            ToolSpec(
                id="search",
                description="Benchmark search override",
                input_schema={"type": "object"},
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    planner = TrialPlanner(dataset_provider=RecordingDatasetProvider())

    planned_trials = planner.plan_benchmark(
        benchmark,
        project_tools=[
            ToolSpec(
                id="search",
                description="Project search",
                input_schema={"type": "object"},
            ),
            ToolSpec(
                id="calculator",
                description="Project calculator",
                input_schema={"type": "object"},
            ),
        ],
    )

    assert len(planned_trials) == 1
    assert [tool.id for tool in planned_trials[0].trial_spec.tools] == [
        "search",
        "calculator",
    ]
    assert {
        tool.id: tool.description for tool in planned_trials[0].trial_spec.tools
    } == {
        "search": "Benchmark search override",
        "calculator": "Project calculator",
    }


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
