"""Evaluate an agent-style engine with scripted turns and first-class tools."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from themis import (
    BenchmarkSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
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
from themis.records import (
    InferenceRecord,
    MessageEvent,
    MessagePayload,
    MetricScore,
    ToolCallEvent,
    ToolCallPayload,
    ToolResultEvent,
    ToolResultPayload,
)
from themis.records.conversation import Conversation
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class AgentDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {
                "item_id": "item-1",
                "question": "What is 6 * 7?",
                "answer": "42",
            }
        ]


class ScriptedAgentEngine:
    def infer(self, trial, context, runtime):
        bootstrap_messages = [
            message.model_dump(mode="json") for message in trial.prompt.messages
        ]
        follow_up_turns = [
            [message.model_dump(mode="json") for message in turn.messages]
            for turn in trial.prompt.follow_up_turns
        ]
        assert [tool.id for tool in trial.tools] == ["calculator"]
        assert trial.tools[0].description == "Benchmark arithmetic tool."
        assert sorted(runtime.tool_handlers) == ["calculator"]
        calculator = runtime.tool_handlers["calculator"]
        tool_result = cast(
            Callable[[object], object],
            calculator,
        )({"expression": "6 * 7"})
        tool_payload = cast(Mapping[str, object], tool_result)
        final_answer = str(tool_payload["value"])
        conversation = Conversation(
            events=[
                MessageEvent(
                    role=PromptRole.ASSISTANT,
                    payload=MessagePayload(
                        content=f"Bootstrap roles: {[message['role'] for message in bootstrap_messages]}"
                    ),
                    event_index=0,
                ),
                ToolCallEvent(
                    role=PromptRole.ASSISTANT,
                    payload=ToolCallPayload(
                        tool_name="calculator",
                        tool_arguments={"expression": "6 * 7"},
                        call_id="tool-1",
                    ),
                    event_index=1,
                ),
                ToolResultEvent(
                    role=PromptRole.TOOL,
                    payload=ToolResultPayload(
                        call_id="tool-1",
                        result={
                            "value": final_answer,
                            "follow_up_turns": follow_up_turns,
                        },
                    ),
                    event_index=2,
                ),
                MessageEvent(
                    role=PromptRole.ASSISTANT,
                    payload=MessagePayload(content=final_answer),
                    event_index=3,
                ),
            ]
        )
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.item_id}",
                raw_text=final_answer,
            ),
            conversation=conversation,
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


def calculator_handler(arguments: object) -> dict[str, str]:
    if not isinstance(arguments, Mapping):
        raise TypeError("calculator handler expected a mapping of arguments")
    expression = arguments.get("expression")
    if expression != "6 * 7":
        raise ValueError(f"unexpected calculator expression: {expression!r}")
    return {"value": "42"}


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("agent-demo", ScriptedAgentEngine())
    registry.register_metric("exact_match", ExactMatchMetric())
    registry.register_tool("calculator", calculator_handler)

    project = ProjectSpec(
        project_name="agent-eval-benchmark",
        researcher_id="examples",
        global_seed=17,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/10-agent-eval-benchmark-first")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
        tools=[
            ToolSpec(
                id="calculator",
                description="Project arithmetic tool.",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            ),
            ToolSpec(
                id="search",
                description="Project search tool.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ],
    )
    benchmark = BenchmarkSpec(
        benchmark_id="agent-eval",
        models=[ModelSpec(model_id="agent-demo-model", provider="agent-demo")],
        slices=[
            SliceSpec(
                slice_id="math-agent",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["agent-default"],
                tool_ids=["calculator"],
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        tools=[
            ToolSpec(
                id="calculator",
                description="Benchmark arithmetic tool.",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                extras={"source": "benchmark"},
            ),
            ToolSpec(
                id="lookup",
                description="Benchmark lookup tool.",
                input_schema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            ),
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="agent-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.SYSTEM,
                        content="You are a careful math agent.",
                    ),
                    PromptMessage(
                        role=PromptRole.DEVELOPER,
                        content="Use tools before you answer.",
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
                                content="Confirm the result for {item.question}.",
                            )
                        ]
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=AgentDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)
    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    candidate_id = trial.candidates[0].candidate_id
    assert candidate_id is not None
    candidate_view = result.view_timeline(candidate_id)
    assert candidate_view is not None

    print(result.aggregate(group_by=["slice_id", "metric_id"]))
    print([event.kind for event in candidate_view.conversation.events])


if __name__ == "__main__":
    main()
