from __future__ import annotations

import pytest

from themis.adapters.openai import openai
from themis.core.contexts import GenerateContext
from themis.core.models import Case, GenerationResult
from themis.core.prompts import FewShotExample, PromptSpec


class FakeUsage:
    input_tokens = 12
    output_tokens = 3


class FakeResponse:
    output_text = "4"
    usage = FakeUsage()
    id = "resp_123"
    headers = {"x-ratelimit-limit-requests": "60"}

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {"id": self.id, "output_text": self.output_text}


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponsesAPI()


@pytest.mark.asyncio
async def test_openai_adapter_generates_results_from_responses_api() -> None:
    client = FakeClient()
    generator = openai("gpt-5.4-mini", client=client, instructions="Answer directly.")
    case = Case(case_id="case-1", input="What is 2+2?", expected_output="4")

    result = await generator.generate(
        case, GenerateContext(run_id="run-1", case_id="case-1", seed=7)
    )

    assert isinstance(result, GenerationResult)
    assert result.candidate_id == "case-1-candidate-7"
    assert result.final_output == "4"
    assert [
        (message.role, message.content) for message in result.conversation or []
    ] == [
        ("system", "Answer directly."),
        ("user", "What is 2+2?"),
        ("assistant", "4"),
    ]
    assert result.token_usage == {"prompt_tokens": 12, "completion_tokens": 3}
    assert result.artifacts == {
        "provider_request_id": "resp_123",
        "raw_response": {"id": "resp_123", "output_text": "4"},
        "response_headers": {"x-ratelimit-limit-requests": "60"},
        "rate_limit": {"requests_per_minute": 60},
    }
    assert client.responses.calls == [
        {
            "model": "gpt-5.4-mini",
            "input": "What is 2+2?",
            "instructions": "Answer directly.",
        }
    ]


def test_openai_adapter_fingerprint_is_deterministic() -> None:
    left = openai("gpt-5.4-mini", instructions="Answer directly.")
    right = openai("gpt-5.4-mini", instructions="Answer directly.")
    changed = openai("gpt-5.4", instructions="Answer directly.")

    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != changed.fingerprint()


@pytest.mark.asyncio
async def test_openai_adapter_can_take_prompt_spec_from_context() -> None:
    client = FakeClient()
    generator = openai("gpt-5.4-mini", client=client)
    case = Case(case_id="case-1", input="What is 2+2?", expected_output="4")

    await generator.generate(
        case,
        GenerateContext(
            run_id="run-1",
            case_id="case-1",
            seed=7,
            prompt_spec=PromptSpec(
                instructions="Answer directly.",
                prefix="Use the examples.",
                few_shot_examples=[
                    FewShotExample(input="1+1", output="2"),
                ],
            ),
        ),
    )

    assert client.responses.calls == [
        {
            "model": "gpt-5.4-mini",
            "input": (
                "Instructions:\nAnswer directly.\n\n"
                "Use the examples.\n\n"
                "Example 1 input:\n1+1\n"
                "Example 1 output:\n2\n\n"
                "Input:\nWhat is 2+2?"
            ),
            "instructions": "Answer directly.",
        }
    ]
