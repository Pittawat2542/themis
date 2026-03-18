"""Use hooks, benchmark prompts, and timeline inspection together."""

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
from themis.contracts.protocols import InferenceResult, RenderedPrompt
from themis.records import CandidateRecord, InferenceRecord, MetricScore
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class SingleItemProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [{"item_id": "item-1", "question": "Explain what you are doing."}]


class _NoOpHook:
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


class InjectSystemPromptHook(_NoOpHook):
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
            )
        )


class ContainsSystemMetric:
    def score(self, trial, candidate, context):
        del trial, context
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="contains_system_prompt",
            value=float("system:Be concise and explicit." in actual),
        )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", PromptAwareEngine())
    registry.register_metric("contains_system_prompt", ContainsSystemMetric())
    registry.register_hook("inject_system", InjectSystemPromptHook(), priority=10)

    project = ProjectSpec(
        project_name="hooks-and-timeline-benchmark",
        researcher_id="examples",
        global_seed=41,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/06-hooks-and-timeline-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="hooked-benchmark",
        models=[ModelSpec(model_id="prompt-aware", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="hooked",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
                scores=[ScoreSpec(name="default", metrics=["contains_system_prompt"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Summarize the request."
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=64)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=SingleItemProvider(),
    )
    result = orchestrator.run_benchmark(benchmark)
    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    candidate_id = trial.candidates[0].candidate_id
    assert candidate_id is not None
    candidate_view = result.view_timeline(candidate_id)
    assert candidate_view is not None
    print(candidate_view.inference.raw_text)


if __name__ == "__main__":
    main()
