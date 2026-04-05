from __future__ import annotations

from types import SimpleNamespace

from themis import Experiment, InMemoryRunStore, get_execution_state
from themis.adapters import openai
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


class _FakeResponses:
    async def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            id="resp-1",
            output_text="4",
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
            headers={"x-ratelimit-limit-requests": "60"},
            model_dump=lambda mode="json": {"request": kwargs, "output_text": "4"},
        )


class _FakeClient:
    @property
    def responses(self) -> _FakeResponses:
        return _FakeResponses()


def run_example() -> dict[str, object]:
    """Execute the OpenAI adapter against a fake injected client."""

    store = InMemoryRunStore()
    generator = openai(
        "gpt-fake",
        client=_FakeClient(),
        instructions="Answer with only the final number.",
    )
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
    result = experiment.run(store=store)
    state = get_execution_state(store, result.run_id)
    candidate = next(iter(state.case_states["case-1"].generated_candidates.values()))
    artifact_keys = [] if candidate.artifacts is None else sorted(candidate.artifacts)
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "artifact_keys": artifact_keys,
    }


if __name__ == "__main__":
    print(run_example())
