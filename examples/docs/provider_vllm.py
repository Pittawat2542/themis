from __future__ import annotations

from types import SimpleNamespace

from themis import Experiment
from themis.adapters import vllm
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset


class _FakeResponses:
    async def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            id="vllm-resp-1",
            output_text="4",
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
            headers={"x-ratelimit-limit-requests": "120"},
            model_dump=lambda mode="json": {"request": kwargs, "output_text": "4"},
        )


class _FakeChatCompletions:
    async def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            id="chat-1",
            choices=[SimpleNamespace(message=SimpleNamespace(content="4"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
            headers={"x-ratelimit-limit-requests": "120"},
            model_dump=lambda mode="json": {
                "request": kwargs,
                "choices": [{"message": {"content": "4"}}],
            },
        )


class _FakeChat:
    @property
    def completions(self) -> _FakeChatCompletions:
        return _FakeChatCompletions()


class _FakeClient:
    @property
    def responses(self) -> _FakeResponses:
        return _FakeResponses()

    @property
    def chat(self) -> _FakeChat:
        return _FakeChat()


def run_example() -> dict[str, object]:
    """Execute the vLLM adapter against a fake injected client."""

    generator = vllm(
        "fake-vllm",
        base_url="http://localhost:8000/v1",
        client=_FakeClient(),
        api_mode="chat_completions",
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
    result = experiment.run()
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "api_mode": generator.api_mode,
    }


if __name__ == "__main__":
    print(run_example())
