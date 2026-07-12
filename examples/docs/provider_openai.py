from __future__ import annotations

from themis.storage import memory_store

from types import SimpleNamespace

from themis import Experiment
from themis.adapters import openai
from themis import Evaluation, Generation
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset


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

    store = memory_store()
    generator = openai(
        "gpt-fake",
        client=_FakeClient(),
        instructions="Answer with only the final number.",
    )
    experiment = Experiment(
        generation=Generation(generator=generator),
        evaluation=Evaluation(),
        datasets=[
            inline_dataset_source(
                Dataset(
                    dataset_id="sample",
                    cases=[Case(case_id="case-1", input={"question": "2+2"})],
                )
            )
        ],
        seeds=[7],
    )
    result = experiment.run(store=store)
    if not result.cases:
        raise RuntimeError("OpenAI example expected at least one case result")
    if not result.cases[0].generated_candidates:
        raise RuntimeError("OpenAI example expected at least one generated candidate")
    candidate = result.cases[0].generated_candidates[0]
    artifact_keys = [] if candidate.artifacts is None else sorted(candidate.artifacts)
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "artifact_keys": artifact_keys,
    }


if __name__ == "__main__":
    print(run_example())
