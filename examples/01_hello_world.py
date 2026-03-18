"""Minimal benchmark-first Themis run."""

from pathlib import Path

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class ArithmeticDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
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


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    project = ProjectSpec(
        project_name="hello-world-benchmark",
        researcher_id="examples",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/01-hello-world-benchmark-first")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="hello-world",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="arithmetic",
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
                        role=PromptRole.USER,
                        content="Question: {item.question}",
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ArithmeticDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)

    for row in result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "source", "prompt_variant_id"]
    ):
        print(row)


if __name__ == "__main__":
    main()
