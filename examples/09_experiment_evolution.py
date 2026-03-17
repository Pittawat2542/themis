"""Evolve a benchmark by adding a model and prompt variant."""

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
        ]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del runtime
        if trial.model.model_id == "baseline" and trial.prompt.id == "baseline":
            answer = context["answer"]
        elif trial.model.model_id == "candidate":
            answer = context["answer"]
        else:
            answer = "wrong"
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.model.model_id}_{trial.item_id}_{trial.prompt.id}",
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


def _baseline() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="evolution-demo",
        models=[ModelSpec(model_id="baseline", provider="demo")],
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
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def _expanded() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="evolution-demo",
        models=[
            ModelSpec(model_id="baseline", provider="demo"),
            ModelSpec(model_id="candidate", provider="demo"),
        ],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline", "cot"],
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
            ),
            PromptVariantSpec(
                id="cot",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content="Think step by step, then answer the problem.",
                    )
                ],
            ),
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    project = ProjectSpec(
        project_name="benchmark-evolution",
        researcher_id="examples",
        global_seed=71,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/09-experiment-evolution-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ArithmeticDatasetProvider(),
    )

    baseline = orchestrator.run_benchmark(_baseline())
    expanded = orchestrator.run_benchmark(_expanded())

    print(
        baseline.aggregate(
            group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
        )
    )
    print(
        expanded.aggregate(
            group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
        )
    )


if __name__ == "__main__":
    main()
