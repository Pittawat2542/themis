from __future__ import annotations

from themis import Experiment
from themis.adapters import langgraph
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


class _FakeGraph:
    async def ainvoke(self, payload: object) -> object:
        return {"answer": "4", "input": payload}

    async def astream_events(self, payload: object, *, version: str):
        del version
        yield {
            "name": "plan",
            "event": "step",
            "data": {"input": payload, "output": {"proposed_answer": "4"}},
        }


def run_example() -> dict[str, object]:
    """Execute the LangGraph adapter against a fake graph."""

    generator = langgraph(_FakeGraph(), graph_id="fake-graph", output_key="answer")
    experiment = Experiment(
        generation=GenerationConfig(generator=generator),
        evaluation=EvaluationConfig(),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="sample",
                cases=[Case(case_id="case-1", input={"question": "2+2"})],
            )
        ],
        seeds=[7],
    )
    result = experiment.run()
    trace_steps = len(result.cases[0].generated_candidates[0].trace or [])
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "trace_steps": trace_steps,
    }


if __name__ == "__main__":
    print(run_example())
