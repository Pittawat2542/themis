"""Run a judge-backed metric through the benchmark-first API."""

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
from themis.specs import DatasetSpec, GenerationSpec, JudgeInferenceSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class JudgeDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [{"item_id": "item-1", "question": "2 + 2", "answer": "4"}]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del runtime
        if trial.model.model_id == "judge-model":
            return InferenceResult(
                inference=InferenceRecord(
                    spec_hash=f"judge_{trial.item_id}",
                    raw_text="PASS",
                )
            )
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.item_id}",
                raw_text="The answer is 4.",
            )
        )


class JudgeBackedMetric:
    def score(self, trial, candidate, context):
        judge = context["judge_service"]
        judge_record = judge.judge(
            "judge_pass",
            candidate,
            JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="demo")
            ),
            trial.prompt.model_copy(
                update={
                    "messages": [
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Judge whether the answer is correct.",
                        )
                    ]
                }
            ),
            context,
        )
        return MetricScore(
            metric_id="judge_pass",
            value=float(judge_record.raw_text == "PASS"),
        )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("judge_pass", JudgeBackedMetric())

    project = ProjectSpec(
        project_name="judge-benchmark",
        researcher_id="examples",
        global_seed=53,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/07-judge-metric-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="judge-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
                scores=[ScoreSpec(name="judge", metrics=["judge_pass"])],
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

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=JudgeDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)
    print(result.aggregate(group_by=["model_id", "slice_id", "metric_id"]))


if __name__ == "__main__":
    main()
