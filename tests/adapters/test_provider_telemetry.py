from __future__ import annotations

from typing import Any, Callable

import pytest

from themis.adapters import (
    anthropic,
    azure_openai,
    bedrock,
    gemini,
    litellm,
    ollama,
)
from themis.adapters._utils import extract_provider_telemetry
from themis.core.contexts import GenerationContext
from themis.core.models import Case, Candidate
from themis.core.workflows import JudgeResponse


class Usage:
    input_tokens = 5
    output_tokens = 1


class AnthropicResponse:
    id = "anthropic_req"
    content = [type("Block", (), {"text": "4"})()]
    usage = Usage()

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {"id": self.id, "content": [{"text": "4"}]}


class AnthropicMessages:
    async def create(self, **kwargs: object) -> AnthropicResponse:
        self.last_call = kwargs
        return AnthropicResponse()


class AnthropicClient:
    def __init__(self) -> None:
        self.messages = AnthropicMessages()


class BedrockClient:
    async def converse(self, **kwargs: object) -> dict[str, object]:
        self.last_call = kwargs
        return {
            "ResponseMetadata": {"RequestId": "bedrock_req"},
            "output": {"message": {"content": [{"text": "4"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 1},
        }


class GeminiModels:
    async def generate_content(self, **kwargs: object) -> dict[str, object]:
        self.last_call = kwargs
        return {
            "response_id": "gemini_req",
            "text": "4",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }


class GeminiClient:
    def __init__(self) -> None:
        self.models = GeminiModels()


class ResponsesAPI:
    async def create(self, **kwargs: object) -> dict[str, object]:
        self.last_call = kwargs
        return {
            "id": "azure_req",
            "output_text": "4",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }


class AzureClient:
    def __init__(self) -> None:
        self.responses = ResponsesAPI()


class OllamaClient:
    async def generate(self, **kwargs: object) -> dict[str, object]:
        self.last_call = kwargs
        return {
            "id": "ollama_req",
            "response": "4",
            "prompt_eval_count": 5,
            "eval_count": 1,
        }


class LiteLLMClient:
    async def acompletion(self, **kwargs: object) -> dict[str, object]:
        self.last_call = kwargs
        return {
            "id": "litellm_req",
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }


def test_extract_provider_telemetry_handles_provider_shapes() -> None:
    telemetry = extract_provider_telemetry(
        {
            "ResponseMetadata": {"RequestId": "bedrock_req"},
            "usage": {"inputTokens": 5, "outputTokens": 1},
        }
    )

    assert telemetry.request_id == "bedrock_req"
    assert telemetry.token_usage == {"prompt_tokens": 5, "completion_tokens": 1}
    assert telemetry.raw_response["ResponseMetadata"] == {"RequestId": "bedrock_req"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "client", "kwargs"),
    [
        (anthropic, AnthropicClient(), {}),
        (bedrock, BedrockClient(), {}),
        (gemini, GeminiClient(), {}),
        (azure_openai, AzureClient(), {"endpoint": "https://example.openai.azure.com"}),
        (ollama, OllamaClient(), {"base_url": "http://localhost:11434"}),
        (litellm, LiteLLMClient(), {}),
    ],
)
async def test_provider_adapters_generate_and_judge_with_shared_telemetry(
    factory: Callable[..., Any], client: object, kwargs: dict[str, object]
) -> None:
    adapter = factory("demo-model", client=client, **kwargs)
    case = Case(case_id="case-1", input="What is 2+2?", expected_output="4")

    generated = await adapter.generate(
        case, GenerationContext(run_id="run-1", case_id="case-1", seed=7)
    )
    judged = await adapter.judge("What is 2+2?", seed=7)

    assert isinstance(generated, Candidate)
    assert generated.final_output == "4"
    assert generated.token_usage == {"prompt_tokens": 5, "completion_tokens": 1}
    assert generated.artifacts is not None
    assert str(generated.artifacts["provider_request_id"]).endswith("_req")
    assert isinstance(judged, JudgeResponse)
    assert judged.raw_response == "4"
    assert judged.token_usage == {"prompt_tokens": 5, "completion_tokens": 1}
