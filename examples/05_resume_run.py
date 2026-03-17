"""Run the same benchmark twice against the same storage root."""

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


class ResumeDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [{"item_id": "item-1", "question": "3 + 4", "answer": "7"}]


class ResumeEngine:
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
            metric_id="exact_match", value=float(actual == context["answer"])
        )


def _benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="resume-math",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
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
                    PromptMessage(role=PromptRole.USER, content="Solve the problem.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ResumeEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    project = ProjectSpec(
        project_name="resume-benchmark",
        researcher_id="examples",
        global_seed=29,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/05-resume-run-benchmark-first")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ResumeDatasetProvider(),
    )

    first = orchestrator.run_benchmark(_benchmark())
    second = orchestrator.run_benchmark(_benchmark())

    print(first.aggregate(group_by=["model_id", "slice_id", "metric_id"]))
    print(second.aggregate(group_by=["model_id", "slice_id", "metric_id"]))


if __name__ == "__main__":
    main()
