"""Run a benchmark comparison with benchmark-native aggregation APIs."""

from pathlib import Path

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
        return [
            {"item_id": "item-1", "question": "1 + 1", "answer": "2"},
            {"item_id": "item-2", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-3", "question": "3 + 3", "answer": "6"},
            {"item_id": "item-4", "question": "4 + 4", "answer": "8"},
        ]


class ComparisonEngine:
    def infer(self, trial, context, runtime):
        del runtime
        if trial.model.model_id == "baseline" and context["item_id"] in {
            "item-2",
            "item-4",
        }:
            answer = "wrong"
        else:
            answer = context["answer"]
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.model.model_id}_{trial.item_id}",
                raw_text=answer,
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match", value=float(actual == context["answer"])
        )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ComparisonEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    project = ProjectSpec(
        project_name="compare-models-benchmark",
        researcher_id="examples",
        global_seed=23,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/04-compare-models-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="paired-math",
        models=[
            ModelSpec(model_id="baseline", provider="demo"),
            ModelSpec(model_id="candidate", provider="demo"),
        ],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dimensions={"source": "synthetic", "format": "qa"},
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Answer the arithmetic problem.",
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

    print(result.paired_compare(metric_id="exact_match", group_by="slice_id"))
    print(result.persist_artifacts(storage_root=project.storage.root_dir))


if __name__ == "__main__":
    main()
