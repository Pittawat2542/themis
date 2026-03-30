from __future__ import annotations

from themis import Experiment, InMemoryRunStore
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset, GenerationResult, Message, TraceStep


class TracedGenerator:
    """Generator example that emits trace and conversation artifacts."""

    component_id = "generator/traced_example"
    version = "1.0"

    def fingerprint(self) -> str:
        return "traced-example-generator"

    async def generate(self, case: Case, ctx: object) -> GenerationResult:
        del ctx
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate",
            final_output={"answer": "4"},
            trace=[
                TraceStep(
                    step_name="reason",
                    step_type="tool",
                    input={"question": case.input},
                    output={"answer": "4"},
                )
            ],
            conversation=[
                Message(role="user", content=case.input),
                Message(role="assistant", content={"answer": "4"}),
            ],
        )


def run_example() -> dict[str, object]:
    """Run with trace-producing generation and inspect the trace view projection."""

    store = InMemoryRunStore()
    experiment = Experiment(
        generation=GenerationConfig(generator=TracedGenerator()),
        evaluation=EvaluationConfig(),
        storage=StorageConfig(store="memory"),
        datasets=[Dataset(dataset_id="sample", cases=[Case(case_id="case-1", input={"question": "2+2"})])],
    )
    result = experiment.run(store=store)
    trace_view = store.get_projection(result.run_id, "trace_view")
    generation_traces = []
    if isinstance(trace_view, dict):
        maybe_traces = trace_view.get("generation_traces")
        if isinstance(maybe_traces, list):
            generation_traces = maybe_traces
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "generation_trace_count": len(generation_traces),
    }


if __name__ == "__main__":
    print(run_example())
