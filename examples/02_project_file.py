"""Run a benchmark using a file-backed project configuration."""

from pathlib import Path

from themis import (
    BenchmarkSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import DatasetSource, PromptRole


class GreetingDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [{"item_id": "item-1", "name": "Themis", "answer": "Hello, Themis!"}]


class GreetingEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=f"Hello, {context['name']}!",
            )
        )


class GreetingMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", GreetingEngine())
    registry.register_metric("exact_match", GreetingMetric())

    project_path = Path(
        ".cache/themis-examples/02-project-file-benchmark-first/project.toml"
    )
    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(
        "\n".join(
            [
                'project_name = "project-file-benchmark"',
                'researcher_id = "examples"',
                "global_seed = 11",
                "",
                "[storage]",
                f'root_dir = "{project_path.parent.as_posix()}"',
                'backend = "sqlite_blob"',
                "store_item_payloads = true",
                'compression = "none"',
                "",
                "[execution_policy]",
                "max_retries = 2",
                "retry_backoff_factor = 1.5",
                "circuit_breaker_threshold = 4",
            ]
        )
        + "\n"
    )

    benchmark = BenchmarkSpec(
        benchmark_id="greetings",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="greeting",
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
                    PromptMessage(
                        role=PromptRole.USER, content="Say hello to {item.name}."
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    orchestrator = Orchestrator.from_project_file(
        str(project_path),
        registry=registry,
        dataset_provider=GreetingDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)

    print(result.aggregate(group_by=["model_id", "slice_id", "metric_id"]))


if __name__ == "__main__":
    main()
