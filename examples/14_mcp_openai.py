"""Run an OpenAI-backed benchmark that exposes a remote MCP server."""

import os
from pathlib import Path

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    McpServerSpec,
    ModelSpec,
    Orchestrator,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    SliceSpec,
    StorageSpec,
)
from themis.catalog import build_catalog_registry
from themis.specs import DatasetSpec, GenerationSpec
from themis.specs.experiment import RuntimeContext
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class McpDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {
                "item_id": "item-1",
                "instruction": "Roll 2d4+1 and reply with only the total.",
            }
        ]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping examples/14_mcp_openai.py because OPENAI_API_KEY is not set.")
        return

    project = ProjectSpec(
        project_name="mcp-openai-example",
        researcher_id="examples",
        global_seed=23,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/14-mcp-openai")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
        mcp_servers=[
            McpServerSpec(
                id="dice",
                server_label="dice",
                server_description="A remote MCP server for dice rolling.",
                server_url="https://dmcp-server.deno.dev/sse",
                allowed_tools=["roll"],
            )
        ],
    )
    benchmark = BenchmarkSpec(
        benchmark_id="mcp-openai",
        models=[ModelSpec(model_id="gpt-5", provider="openai")],
        slices=[
            SliceSpec(
                slice_id="dice-roll",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=23),
                prompt_variant_ids=["mcp-default"],
                generation=GenerationSpec(),
                mcp_server_ids=["dice"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="mcp-default",
                family="agent",
                messages=[
                    PromptMessage(
                        role=PromptRole.SYSTEM,
                        content="Use the available MCP tools when needed.",
                    ),
                    PromptMessage(
                        role=PromptRole.USER,
                        content="{item.instruction}",
                    ),
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=build_catalog_registry(["openai"]),
        dataset_provider=McpDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark, runtime=RuntimeContext())
    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    candidate = trial.candidates[0]
    print(candidate.inference.raw_text)
    if candidate.conversation is not None:
        print([event.kind for event in candidate.conversation.events])


if __name__ == "__main__":
    main()
