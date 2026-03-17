"""Use a custom parser and metric with the benchmark-first API."""

import re
from pathlib import Path

from themis import (
    BenchmarkSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    ParseSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import ExtractionRecord, InferenceRecord, MetricScore
from themis.records import CandidateRecord
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class FactsDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {"item_id": "item-1", "question": "What is 6 * 7?", "answer": "42"},
            {"item_id": "item-2", "question": "What is 8 * 8?", "answer": "64"},
        ]


class VerboseAnswerEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=f"I checked carefully. The answer is {context['answer']}.",
            )
        )


class NumberExtractor:
    def extract(self, trial, candidate: CandidateRecord, config=None):
        del trial, config
        text = candidate.inference.raw_text if candidate.inference else ""
        match = re.search(r"(\d+)", text or "")
        return ExtractionRecord(
            spec_hash=f"ext_{candidate.spec_hash}",
            extractor_id="number_extractor",
            success=match is not None,
            parsed_answer=match.group(1) if match else None,
            failure_reason=None if match else "No number found.",
        )


class ParsedExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = extraction.parsed_answer if extraction is not None else None
        return MetricScore(
            metric_id="parsed_exact_match",
            value=float(parsed == context["answer"]),
            details={"parsed": parsed, "expected": context["answer"]},
        )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", VerboseAnswerEngine())
    registry.register_extractor("number_extractor", NumberExtractor())
    registry.register_metric("parsed_exact_match", ParsedExactMatchMetric())

    project = ProjectSpec(
        project_name="custom-parser-benchmark",
        researcher_id="examples",
        global_seed=17,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/03-custom-extractor-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="custom-parser",
        models=[ModelSpec(model_id="verbose-answer-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
                parses=[ParseSpec(name="parsed", extractors=["number_extractor"])],
                scores=[
                    ScoreSpec(
                        name="parsed-score",
                        parse="parsed",
                        metrics=["parsed_exact_match"],
                    )
                ],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                family="qa",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="Answer the question.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=FactsDatasetProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)

    for row in result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
    ):
        print(row)


if __name__ == "__main__":
    main()
