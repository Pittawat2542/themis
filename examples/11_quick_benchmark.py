"""BenchmarkSpec.simple() and BenchmarkSpec.preview() — minimal setup patterns.

Demonstrates:
- BenchmarkSpec.simple() for quick single-model exploration
- BenchmarkSpec.preview() for inspecting rendered prompts without running inference
- PluginRegistry.from_dict() to reduce registration boilerplate
"""

from pathlib import Path

from themis import (
    BenchmarkSpec,
    ExecutionPolicySpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.types.enums import CompressionCodec, DatasetSource


class ArithmeticProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "3 + 3", "answer": "6"},
        ]


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
    # ── 1. BenchmarkSpec.simple() ──────────────────────────────────────────────
    # One call constructs a complete, valid BenchmarkSpec.  Great for exploration
    # before you need multiple models, prompt variants, or scoring pipelines.
    benchmark = BenchmarkSpec.simple(
        benchmark_id="arithmetic-quick",
        model_id="demo-model",
        provider="demo",
        dataset_source=DatasetSource.MEMORY,
        dataset_id="arithmetic",
        prompt="What is {item.question}?",
        metric="exact_match",
    )

    # ── 2. BenchmarkSpec.preview() ────────────────────────────────────────────
    # Render all prompt variants against a sample item — no inference, no storage.
    sample_item = {"question": "2 + 2", "answer": "4"}
    previewed = benchmark.preview(sample_item)
    print("=== Prompt preview ===")
    for entry in previewed:
        print(f"variant: {entry['prompt_variant_id']}")
        for msg in entry["messages"]:
            print(f"  [{msg['role']}] {msg['content']}")
    print()

    # ── 3. PluginRegistry.from_dict() ─────────────────────────────────────────
    # Declare all plugins in one mapping instead of calling register_* individually.
    registry = PluginRegistry.from_dict(
        {
            "engines": {"demo": DemoEngine()},
            "metrics": {"exact_match": ExactMatchMetric()},
        }
    )

    project = ProjectSpec(
        project_name="quick-benchmark",
        researcher_id="examples",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/11-quick-benchmark")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ArithmeticProvider(),
    )

    result = orchestrator.run_benchmark(benchmark)
    print("=== Results ===")
    for row in result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
    ):
        print(row)


if __name__ == "__main__":
    main()
