from __future__ import annotations

from pathlib import Path

import pytest

from pydantic import ValidationError

from themis import (
    BenchmarkResult,
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    McpServerSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTurnSpec,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
    ToolSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.contracts.protocols import RenderedPrompt
from themis.orchestration.run_manifest import RunHandle
from themis.records import InferenceRecord, MetricScore
from themis.registry.plugin_registry import EngineCapabilities
from themis.specs.experiment import RuntimeContext
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class DemoDatasetProvider:
    def scan(self, slice_spec, query):
        del query
        return [{"item_id": "item-1", "question": "2 + 2", "answer": "4"}]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=str(context["answer"]),
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


def test_orchestrator_renders_benchmark_prompt_before_inference_and_preserves_prompt_metadata(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class RenderingEngine:
        def infer(self, trial, context, runtime):
            del context, runtime
            seen["content"] = trial.prompt.messages[0].content
            seen["family"] = trial.prompt.family
            seen["variables"] = trial.prompt.variables
            return InferenceResult(
                inference=InferenceRecord(
                    spec_hash="inf_rendered",
                    raw_text="4",
                )
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", RenderingEngine())
    registry.register_metric("exact_match", RenderingMetric())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                variables={"tone": "formal"},
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content=(
                            "Solve: {item.question} "
                            "[{prompt.family}/{prompt.variables[tone]}] "
                            "{slice.dimensions[source]} "
                            "{runtime.run_labels[phase]}"
                        ),
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(
        benchmark,
        runtime=RuntimeContext(run_labels={"phase": "smoke"}),
    )

    assert isinstance(result, BenchmarkResult)
    assert seen == {
        "content": "Solve: 2 + 2 [qa/formal] synthetic smoke",
        "family": "qa",
        "variables": {"tone": "formal"},
    }


def test_orchestrator_runs_benchmark_and_returns_benchmark_result(
    tmp_path: Path,
) -> None:
    project = ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
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
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(benchmark)

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_id == "demo-benchmark"
    assert result.slice_ids == ["qa"]
    assert result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "source", "prompt_variant_id"]
    ) == [
        {
            "count": 1,
            "mean": 1.0,
            "metric_id": "exact_match",
            "model_id": "demo-model",
            "prompt_variant_id": "qa-default",
            "slice_id": "qa",
            "source": "synthetic",
        }
    ]


def test_orchestrator_renders_bootstrap_and_follow_up_turns_for_agent_prompts(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class AgentEngine:
        def infer(self, trial, context, runtime):
            del context, runtime
            seen["bootstrap"] = [
                (message.role.value, message.content)
                for message in trial.prompt.messages
            ]
            seen["follow_up_turns"] = [
                [(message.role.value, message.content) for message in turn.messages]
                for turn in trial.prompt.follow_up_turns
            ]
            return InferenceResult(
                inference=InferenceRecord(
                    spec_hash="inf_rendered_agent",
                    raw_text="4",
                )
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = _build_project(tmp_path)
    registry = PluginRegistry()
    registry.register_inference_engine("demo", AgentEngine())
    registry.register_metric("exact_match", RenderingMetric())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="agent-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic"},
                prompt_variant_ids=["agent-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.DEVELOPER,
                        content="Follow {prompt.family}.",
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
                                content="Double check in {runtime.run_labels[phase]} mode.",
                            )
                        ]
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(
        benchmark,
        runtime=RuntimeContext(run_labels={"phase": "turn-2"}),
    )

    assert isinstance(result, BenchmarkResult)
    assert seen == {
        "bootstrap": [
            ("developer", "Follow agent."),
            ("user", "Solve: 2 + 2"),
        ],
        "follow_up_turns": [[("user", "Double check in turn-2 mode.")]],
    }


def test_orchestrator_merges_project_and_benchmark_tools_and_injects_handlers(
    tmp_path: Path,
) -> None:
    search_handler = object()
    calculator_handler = object()
    seen: dict[str, object] = {}

    class ToolAwareEngine:
        def infer(self, trial, context, runtime):
            del context
            seen["tool_descriptions"] = {
                tool.id: tool.description for tool in trial.tools
            }
            seen["tool_handler_keys"] = sorted(runtime.tool_handlers)
            seen["tool_handler_identity"] = runtime.tool_handlers["search"]
            return InferenceResult(
                inference=InferenceRecord(spec_hash="inf_tooling", raw_text="4")
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = _build_project(tmp_path).model_copy(
        update={
            "tools": [
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
            ]
        }
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ToolAwareEngine())
    registry.register_metric("exact_match", RenderingMetric())
    registry.register_tool("search", search_handler)
    registry.register_tool("calculator", calculator_handler)

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="tool-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                tool_ids=["search", "calculator"],
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
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
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(benchmark)

    assert isinstance(result, BenchmarkResult)
    assert seen["tool_descriptions"] == {
        "search": "Benchmark search override",
        "calculator": "Project calculator",
    }
    assert seen["tool_handler_keys"] == ["calculator", "search"]
    assert seen["tool_handler_identity"] is search_handler


def test_pre_inference_hook_can_mutate_tools_without_losing_follow_up_turns(
    tmp_path: Path,
) -> None:
    search_handler = object()
    calculator_handler = object()
    seen: dict[str, object] = {}

    class FilteringHook:
        def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
            del trial
            seen["hook_tool_ids"] = [tool.id for tool in prompt.tools]
            seen["hook_follow_up_turns"] = len(prompt.follow_up_turns)
            return prompt.model_copy(
                update={"tools": [tool for tool in prompt.tools if tool.id == "search"]}
            )

        def post_inference(self, trial, result):
            return result

        def pre_extraction(self, trial, candidate):
            return candidate

        def post_extraction(self, trial, candidate):
            return candidate

        def pre_eval(self, trial, candidate):
            return candidate

        def post_eval(self, trial, candidate):
            return candidate

    class ToolAwareEngine:
        def infer(self, trial, context, runtime):
            del context
            seen["engine_tool_ids"] = [tool.id for tool in trial.tools]
            seen["engine_follow_up_turns"] = len(trial.prompt.follow_up_turns)
            seen["engine_tool_handler_keys"] = sorted(runtime.tool_handlers)
            return InferenceResult(
                inference=InferenceRecord(spec_hash="inf_tool_filter", raw_text="4")
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = _build_project(tmp_path).model_copy(
        update={
            "tools": [
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
            ]
        }
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ToolAwareEngine())
    registry.register_metric("exact_match", RenderingMetric())
    registry.register_hook("filtering", FilteringHook())
    registry.register_tool("search", search_handler)
    registry.register_tool("calculator", calculator_handler)

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="tool-hook-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                tool_ids=["search", "calculator"],
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
                follow_up_turns=[
                    PromptTurnSpec(
                        messages=[
                            PromptMessage(
                                role=PromptRole.USER,
                                content="Double check the answer.",
                            )
                        ]
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(benchmark)

    assert isinstance(result, BenchmarkResult)
    assert seen["hook_tool_ids"] == ["search", "calculator"]
    assert seen["hook_follow_up_turns"] == 1
    assert seen["engine_tool_ids"] == ["search"]
    assert seen["engine_follow_up_turns"] == 1
    assert seen["engine_tool_handler_keys"] == ["search"]


def test_pre_inference_hook_can_mutate_mcp_servers_without_losing_tools(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class FilteringHook:
        def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
            del trial
            seen["hook_mcp_server_ids"] = [server.id for server in prompt.mcp_servers]
            seen["hook_tool_ids"] = [tool.id for tool in prompt.tools]
            return prompt.model_copy(
                update={
                    "mcp_servers": [
                        server for server in prompt.mcp_servers if server.id == "dice"
                    ]
                }
            )

        def post_inference(self, trial, result):
            return result

        def pre_extraction(self, trial, candidate):
            return candidate

        def post_extraction(self, trial, candidate):
            return candidate

        def pre_eval(self, trial, candidate):
            return candidate

        def post_eval(self, trial, candidate):
            return candidate

    class McpAwareEngine:
        def infer(self, trial, context, runtime):
            del context, runtime
            seen["engine_mcp_server_ids"] = [server.id for server in trial.mcp_servers]
            seen["engine_tool_ids"] = [tool.id for tool in trial.tools]
            return InferenceResult(
                inference=InferenceRecord(spec_hash="inf_mcp_hook", raw_text="4")
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = _build_project(tmp_path).model_copy(
        update={
            "tools": [
                ToolSpec(
                    id="search",
                    description="Project search",
                    input_schema={"type": "object"},
                )
            ],
            "mcp_servers": [
                McpServerSpec(
                    id="dice",
                    server_label="dice",
                    server_url="https://dmcp-server.deno.dev/sse",
                )
            ],
        }
    )
    registry = PluginRegistry()
    registry.register_inference_engine(
        "openai",
        McpAwareEngine(),
        capabilities=EngineCapabilities(supports_mcp=True),
    )
    registry.register_metric("exact_match", RenderingMetric())
    registry.register_hook("filtering", FilteringHook())
    registry.register_tool("search", object())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="tool-hook-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                tool_ids=["search"],
                mcp_server_ids=["dice", "calendar"],
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        mcp_servers=[
            McpServerSpec(
                id="calendar",
                server_label="google_calendar",
                connector_id="connector_googlecalendar",
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(benchmark)

    assert isinstance(result, BenchmarkResult)
    assert seen["hook_mcp_server_ids"] == ["dice", "calendar"]
    assert seen["hook_tool_ids"] == ["search"]
    assert seen["engine_mcp_server_ids"] == ["dice"]
    assert seen["engine_tool_ids"] == ["search"]


def _build_project(tmp_path: Path) -> ProjectSpec:
    return ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def _build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
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
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )


def _build_orchestrator(tmp_path: Path) -> Orchestrator:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())
    return Orchestrator.from_project_spec(
        _build_project(tmp_path),
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )


def test_orchestrator_submit_resume_and_plan_are_benchmark_native(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    benchmark = _build_benchmark()

    manifest = orchestrator.plan(benchmark)
    handle = orchestrator.submit(benchmark, runtime=RuntimeContext())
    resumed = orchestrator.resume(handle.run_id)

    assert manifest.benchmark_spec is not None
    assert manifest.benchmark_spec.benchmark_id == "demo-benchmark"
    assert isinstance(handle, RunHandle)
    assert handle.status == "completed"
    assert isinstance(resumed, BenchmarkResult)
    assert resumed.benchmark_id == "demo-benchmark"
    assert resumed.slice_ids == ["qa"]
    stored_manifest = orchestrator._run_planning.manifest_repo.get_manifest(
        handle.run_id
    )
    assert stored_manifest is not None
    assert stored_manifest.source_kind == "benchmark"
    assert stored_manifest.benchmark_spec is not None
    assert stored_manifest.benchmark_spec.benchmark_id == "demo-benchmark"


def test_orchestrator_diff_specs_supports_benchmark_source_diffs(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    baseline = _build_benchmark()
    treatment = baseline.model_copy(
        update={
            "models": [
                *baseline.models,
                ModelSpec(model_id="demo-model-2", provider="demo"),
            ]
        }
    )

    diff = orchestrator.diff_specs(baseline, treatment)

    assert diff.source_kind == "benchmark"
    assert "models" in diff.changed_source_fields
    assert diff.added_trial_hashes
    assert diff.removed_trial_hashes == []


def test_orchestrator_diff_specs_rehashes_trials_for_query_and_render_fields(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    baseline = _build_benchmark()
    treatment_slice = baseline.slices[0].model_copy(
        update={
            "dataset_query": DatasetQuerySpec.subset(1, seed=11),
            "dimensions": {"source": "synthetic", "format": "cot"},
        }
    )
    treatment_prompt = baseline.prompt_variants[0].model_copy(
        update={"variables": {"tone": "formal"}}
    )
    treatment = baseline.model_copy(
        update={
            "slices": [treatment_slice],
            "prompt_variants": [treatment_prompt],
        }
    )

    diff = orchestrator.diff_specs(baseline, treatment)

    assert "slices" in diff.changed_source_fields
    assert "prompt_variants" in diff.changed_source_fields
    assert diff.added_trial_hashes
    assert diff.removed_trial_hashes


def test_benchmark_spec_rejects_duplicate_slice_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate slice_id"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                ),
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                ),
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )


def test_benchmark_spec_rejects_duplicate_prompt_variant_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate prompt variant"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                ),
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Score: {item.question}",
                        )
                    ],
                ),
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )


def test_benchmark_spec_rejects_slices_with_unknown_prompt_variant_ids() -> None:
    with pytest.raises(ValidationError, match="missing-variant"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    prompt_variant_ids=["missing-variant"],
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )
