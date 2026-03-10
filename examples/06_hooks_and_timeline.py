"""Use hooks, telemetry, and timeline inspection together."""

from pathlib import Path

from themis import (
    DatasetSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
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
from themis.contracts.protocols import InferenceResult, RenderedPrompt
from themis.records.conversation import Conversation, MessageEvent, MessagePayload
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord
from themis.telemetry.bus import TelemetryBus


class SingleItemLoader:
    def load_task_items(self, task):
        del task
        return [{"item_id": "item-1", "question": "Explain what you are doing."}]


class InjectSystemPromptHook:
    """Prepends a system message before the engine sees the prompt."""

    def pre_inference(self, trial, prompt: RenderedPrompt) -> RenderedPrompt:
        del trial
        messages = [PromptMessage(role="system", content="Be concise and explicit.")]
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
                        role="assistant",
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
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/06-hooks-and-timeline")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="prompt-aware", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="hooked-task",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["contains_system_prompt"],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[PromptMessage(role="user", content="Summarize the request.")],
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
    trial = result.get_trial(trial_hash)
    candidate_id = trial.candidates[0].candidate_id
    candidate_view = result.view_timeline(candidate_id)

    print("Telemetry events:", ", ".join(sorted(set(seen_events))))
    print(
        "Trial stages:", ", ".join(stage.stage for stage in trial_view.timeline.stages)
    )
    print("Candidate output:", candidate_view.inference.raw_text)


if __name__ == "__main__":
    main()
