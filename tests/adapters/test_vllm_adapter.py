from __future__ import annotations

import pytest

from themis.adapters.vllm import vllm
from themis.core.contexts import GenerateContext
from themis.core.models import Case


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "FakeResponse",
            (),
            {
                "output_text": "4",
                "usage": type("Usage", (), {"input_tokens": 4, "output_tokens": 1})(),
                "id": "resp_vllm",
                "headers": {"server": "vllm"},
                "model_dump": lambda self, mode="json": {"output_text": "4", "id": "resp_vllm"},
            },
        )()


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = type("Choice", (), {"message": type("Message", (), {"content": "4"})()})()
        usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 1})()
        return type(
            "FakeChatResponse",
            (),
            {
                "choices": [choice],
                "usage": usage,
                "id": "chatcmpl_vllm",
                "headers": {"server": "vllm"},
                "model_dump": lambda self, mode="json": {"id": "chatcmpl_vllm", "choices": [{"message": {"content": "4"}}]},
            },
        )()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponsesAPI()
        self.chat = type("ChatAPI", (), {"completions": FakeChatCompletionsAPI()})()


@pytest.mark.asyncio
async def test_vllm_adapter_supports_openai_compatible_responses_mode() -> None:
    client = FakeClient()
    generator = vllm("qwen2.5", base_url="http://localhost:8000/v1", client=client)

    result = await generator.generate(
        Case(case_id="case-1", input="What is 2+2?", expected_output="4"),
        GenerateContext(run_id="run-1", case_id="case-1", seed=7),
    )

    assert result.final_output == "4"
    assert result.token_usage == {"prompt_tokens": 4, "completion_tokens": 1}
    assert client.responses.calls == [{"model": "qwen2.5", "input": "What is 2+2?"}]


@pytest.mark.asyncio
async def test_vllm_adapter_supports_chat_completions_mode() -> None:
    client = FakeClient()
    generator = vllm(
        "qwen2.5",
        base_url="http://localhost:8000/v1",
        client=client,
        api_mode="chat_completions",
    )

    result = await generator.generate(
        Case(case_id="case-1", input="What is 2+2?", expected_output="4"),
        GenerateContext(run_id="run-1", case_id="case-1", seed=7),
    )

    assert result.final_output == "4"
    assert client.chat.completions.calls == [
        {"model": "qwen2.5", "messages": [{"role": "user", "content": "What is 2+2?"}]}
    ]
