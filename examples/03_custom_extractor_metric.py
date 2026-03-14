"""Use a custom extractor and metric to score structured output."""

import re
from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    OutputTransformSpec,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTemplateSpec,
    StorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import ExtractionRecord, InferenceRecord, MetricScore


class FactsDatasetLoader:
    def load_task_items(self, task):
        del task
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
                raw_text=f"I checked my work carefully. The answer is {context['answer']}.",
            )
        )


class NumberExtractor:
    def extract(self, trial, candidate, config):
        del trial, config
        text = candidate.inference.raw_text if candidate.inference else ""
        match = re.search(r"(\d+)", text)
        return ExtractionRecord(
            spec_hash=f"ext_{candidate.spec_hash}",
            extractor_id="number_extractor",
            success=match is not None,
            parsed_answer=match.group(1) if match else None,
            failure_reason=None if match else "No number found in the model output.",
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


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", VerboseAnswerEngine())
    registry.register_extractor("number_extractor", NumberExtractor())
    registry.register_metric("parsed_exact_match", ParsedExactMatchMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="custom-extractor",
        researcher_id="examples",
        global_seed=17,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/03-custom-extractor")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="verbose-answer-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="math-with-extraction",
                dataset=DatasetSpec(source="memory"),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="parsed",
                        extractor_chain=ExtractorChainSpec(
                            extractors=[ExtractorRefSpec(id="number_extractor")]
                        ),
                    )
                ],
                evaluations=[
                    EvaluationSpec(
                        name="parsed-score",
                        transform="parsed",
                        metrics=["parsed_exact_match"],
                    )
                ],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[PromptMessage(role="user", content="Answer the question.")],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )


def main() -> None:
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=FactsDatasetLoader(),
    )
    result = orchestrator.run(build_experiment())
    transform_result = result.for_transform(result.transform_hashes[0])

    for trial in result.iter_trials():
        candidate = trial.candidates[0]
        transform_trial = transform_result.get_trial(trial.spec_hash)
        if transform_trial is None:
            raise RuntimeError(f"Missing transform overlay for {trial.spec_hash}.")
        extraction = transform_trial.candidates[0].best_extraction()
        if extraction is None:
            raise RuntimeError(f"Missing extraction for {trial.spec_hash}.")
        score = candidate.evaluation.aggregate_scores["parsed_exact_match"]
        print(
            f"{trial.trial_spec.item_id}: parsed={extraction.parsed_answer!r} "
            f"score={score:.1f}"
        )


if __name__ == "__main__":
    main()
