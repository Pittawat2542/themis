"""Run paired model comparisons and export a report.

Requires the optional stats extra:

    uv add "themis-eval[stats]"
"""

from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTemplateSpec,
    StorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.types.enums import PromptRole, DatasetSource, CompressionCodec


class ArithmeticDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "1 + 1", "answer": "2"},
            {"item_id": "item-2", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-3", "question": "3 + 3", "answer": "6"},
            {"item_id": "item-4", "question": "4 + 4", "answer": "8"},
            {"item_id": "item-5", "question": "5 + 5", "answer": "10"},
            {"item_id": "item-6", "question": "6 + 6", "answer": "12"},
        ]


class ComparisonEngine:
    """Makes the candidate model consistently stronger than the baseline."""

    def infer(self, trial, context, runtime):
        del runtime
        if trial.model.model_id == "baseline" and context["item_id"] in {
            "item-2",
            "item-4",
            "item-6",
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
            metric_id="exact_match",
            value=float(actual == context["answer"]),
            details={"actual": actual, "expected": context["answer"]},
        )


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ComparisonEngine())
    registry.register_metric("exact_match", ExactMatchMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="compare-models",
        researcher_id="examples",
        global_seed=23,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/04-compare-models")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[
            ModelSpec(model_id="baseline", provider="demo"),
            ModelSpec(model_id="candidate", provider="demo"),
        ],
        tasks=[
            TaskSpec(
                task_id="paired-math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Answer the arithmetic problem."
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def _format_display_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=ArithmeticDatasetLoader(),
    )
    result = orchestrator.run(build_experiment())

    comparison = result.compare(
        metric_id="exact_match",
        baseline_model_id="baseline",
        treatment_model_id="candidate",
        p_value_correction="holm",
    )
    row = comparison.rows[0]

    report_path = Path(".cache/themis-examples/04-compare-models/report.md")
    builder = result.report()
    builder.build(p_value_correction="holm")
    builder.to_markdown(str(report_path))

    print(
        "delta_mean=",
        round(row.delta_mean, 3),
        "adjusted_p_value=",
        round(row.adjusted_p_value, 6),
        "pairs=",
        row.pair_count,
    )
    print("Report written to:", _format_display_path(report_path))


if __name__ == "__main__":
    main()
