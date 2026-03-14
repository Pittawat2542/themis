"""Run a judge-backed metric and inspect the stored audit trail.

Requires the optional compression extra because judge audits are stored as
artifacts when the project uses compressed storage:

    uv add "themis-eval[compression]"
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
from themis.specs.foundational import JudgeInferenceSpec


class SingleMathLoader:
    def load_task_items(self, task):
        del task
        return [{"item_id": "item-1", "question": "6 * 7", "answer": "42"}]


class CandidateEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=f"The answer is {context['answer']}.",
            )
        )


class JudgeEngine:
    def infer(self, trial, context, runtime):
        del trial, context, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash="judge_inf_item_1",
                raw_text="PASS",
            )
        )


class JudgeBackedMetric:
    def score(self, trial, candidate, context):
        judge = context["judge_service"]
        judge_inference = judge.judge(
            metric_id="judge_pass",
            parent_candidate=candidate,
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="judge"),
            ),
            prompt=PromptTemplateSpec(
                id="judge-prompt",
                messages=[
                    PromptMessage(
                        role="user", content="Does the answer match the reference?"
                    )
                ],
            ),
            runtime={"task_spec": trial.task, "dataset_context": context},
        )
        return MetricScore(
            metric_id="judge_pass",
            value=float(judge_inference.raw_text == "PASS"),
            details={"judge_raw_text": judge_inference.raw_text},
        )


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("candidate", CandidateEngine())
    registry.register_inference_engine("judge", JudgeEngine())
    registry.register_metric("judge_pass", JudgeBackedMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="judge-metric",
        researcher_id="examples",
        global_seed=53,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/07-judge-metric")),
            compression="zstd",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="candidate-model", provider="candidate")],
        tasks=[
            TaskSpec(
                task_id="judge-task",
                dataset=DatasetSpec(source="memory"),
                generation=GenerationSpec(),
                evaluations=[EvaluationSpec(name="default", metrics=["judge_pass"])],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(role="user", content="Solve the arithmetic problem.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )


def main() -> None:
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=SingleMathLoader(),
    )
    result = orchestrator.run(build_experiment())

    trial = result.get_trial(result.trial_hashes[0])
    candidate_id = trial.candidates[0].candidate_id
    candidate_view = result.view_timeline(candidate_id)

    print(
        "Metric score:", trial.candidates[0].evaluation.aggregate_scores["judge_pass"]
    )
    print("Judge audit calls:", len(candidate_view.judge_audit.judge_calls))
    print(
        "Judge raw text:", candidate_view.judge_audit.judge_calls[0].inference.raw_text
    )


if __name__ == "__main__":
    main()
