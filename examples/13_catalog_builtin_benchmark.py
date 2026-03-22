"""Run a built-in catalog benchmark through the public themis.catalog helpers."""

from pathlib import Path

from themis import Orchestrator
from themis.catalog import build_catalog_benchmark_project, list_catalog_benchmarks


def load_fixture_rows(dataset_id: str, split: str, revision: str | None):
    """Offline dataset fixture so the example stays runnable without network access."""

    del revision
    if dataset_id != "TIGER-Lab/MMLU-Pro":
        raise ValueError(f"Unexpected dataset_id: {dataset_id}")
    if split != "test":
        raise ValueError(f"Unexpected split: {split}")
    return [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "answer_index": 1,
            "category": "astronomy",
            "src": "fixture",
        }
    ]


def main() -> None:
    print("=== Built-in benchmark ids ===")
    print(", ".join(list_catalog_benchmarks()))
    print()

    project, benchmark, registry, dataset_provider, definition = (
        build_catalog_benchmark_project(
            benchmark_id="mmlu_pro",
            model_id="demo-model",
            provider="demo",
            storage_root=Path(".cache/themis-examples/13-catalog-builtin-benchmark"),
            huggingface_loader=load_fixture_rows,
        )
    )

    print("=== Prompt preview ===")
    for entry in definition.render_preview(model_id="demo-model", provider="demo"):
        print(f"variant: {entry['prompt_variant_id']}")
        for message in entry["messages"]:
            print(f"  [{message['role']}] {message['content']}")
    print()

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=dataset_provider,
    )
    result = orchestrator.run_benchmark(benchmark)

    print("=== Results ===")
    for row in result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
    ):
        print(row)


if __name__ == "__main__":
    main()
