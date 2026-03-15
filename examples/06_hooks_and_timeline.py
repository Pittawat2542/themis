"""Use hooks, telemetry, and timeline inspection together."""

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
    SqliteBlobStorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult, RenderedPrompt
from themis.records import (
    CandidateRecord,
    Conversation,
    InferenceRecord,
    MessageEvent,
    MessagePayload,
    MetricScore,
)
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import PromptRole, DatasetSource, CompressionCodec


class SingleItemLoader:
    def load_task_items(self, task):
        del task
        return [{"item_id": "item-1", "question": "Explain what you are doing."}]


class _NoOpPipelineHook:
    def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
        del trial
        return prompt

    def post_inference(self, trial, result: InferenceResult) -> InferenceResult:
        del trial
        return result

    def pre_extraction(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        del trial
        return candidate

    def post_extraction(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        del trial
        return candidate

    def pre_eval(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        del trial
        return candidate

    def post_eval(self, trial, candidate: CandidateRecord) -> CandidateRecord:
        del trial
        return candidate


class InjectSystemPromptHook(_NoOpPipelineHook):
    """Prepends a system message before the engine sees the prompt."""

    def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
        del trial
        messages = [
            PromptMessage(role=PromptRole.SYSTEM, content="Be concise and explicit.")
        ]
        messages.extend(prompt.messages)
        return prompt.model_copy(update={"messages": messages})


class PromptAwareEngine:
    def infer(self, trial, context, runtime):
        del runtime
        rendered_prompt = [
            f"{message.role}:{message.content}" for message in trial.prompt.messages
        ]
        answer = f"{rendered_prompt[0]} | question={context['question']}"
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=answer,
            ),
            conversation=Conversation(
                events=[
                    MessageEvent(
                        role=PromptRole.ASSISTANT,
                        event_index=0,
                        payload=MessagePayload(content=answer),
                    )
                ]
            ),
        )


class ContainsSystemMetric:
    def score(self, trial, candidate, context):
        del trial, context
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="contains_system_prompt",
            value=float("system:Be concise and explicit." in actual),
            details={"actual": actual},
        )


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", PromptAwareEngine())
    registry.register_metric("contains_system_prompt", ContainsSystemMetric())
    registry.register_hook("inject_system", InjectSystemPromptHook(), priority=10)
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="hooks-and-timeline",
        researcher_id="examples",
        global_seed=41,
        storage=SqliteBlobStorageSpec(
            root_dir=str(Path(".cache/themis-examples/06-hooks-and-timeline")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="prompt-aware", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="hooked-task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                evaluations=[
                    EvaluationSpec(
                        name="default",
                        metrics=["contains_system_prompt"],
                    )
                ],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Summarize the request."
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )


def main() -> None:
    telemetry_bus = TelemetryBus()
    seen_events: list[str] = []
    telemetry_bus.subscribe(lambda event: seen_events.append(event.name))

    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=SingleItemLoader(),
        telemetry_bus=telemetry_bus,
    )
    result = orchestrator.run(build_experiment())

    trial_hash = result.trial_hashes[0]
    trial_view = result.view_timeline(trial_hash, record_type="trial")
    assert trial_view is not None
    assert trial_view.timeline is not None

    trial = result.get_trial(trial_hash)
    assert trial is not None
    candidate_id = trial.candidates[0].candidate_id
    assert candidate_id is not None
    candidate_view = result.view_timeline(candidate_id)
    assert candidate_view is not None
    assert candidate_view.inference is not None

    print("Telemetry events:", ", ".join(sorted(set(seen_events))))
    print(
        "Trial stages:", ", ".join(stage.stage for stage in trial_view.timeline.stages)
    )
    print("Candidate output:", candidate_view.inference.raw_text)


if __name__ == "__main__":
    main()
