"""OpenAI Responses API generator adapter."""

from __future__ import annotations

from typing import Any, Protocol, cast

from themis.adapters._utils import (
    dump_response,
    extract_headers,
    extract_rate_limit,
    extract_token_usage,
    stable_fingerprint,
)
from themis.core.contexts import GenerateContext
from themis.core.models import Case, GenerationResult, Message


class _ResponsesCreateAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _OpenAIResponsesClient(Protocol):
    @property
    def responses(self) -> _ResponsesCreateAPI: ...


class OpenAIGenerator:
    component_id = "generator/openai"
    version = "1.0"

    def __init__(
        self,
        model_id: str,
        *,
        client: _OpenAIResponsesClient | None = None,
        instructions: str | None = None,
        input_builder: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client
        self.instructions = instructions
        self.input_builder = input_builder
        self.base_url = base_url
        self.api_key = api_key
        self.provider_key = f"openai:{(base_url or 'https://api.openai.com/v1').rstrip('/')}"

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "model_id": self.model_id,
                "instructions": self.instructions,
                "base_url": self.base_url,
            }
        )

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        client = self._client or self._build_client()
        request_input = self.input_builder(case) if self.input_builder is not None else case.input
        payload: dict[str, object] = {"model": self.model_id, "input": request_input}
        if self.instructions is not None:
            payload["instructions"] = self.instructions

        response = await client.responses.create(**payload)
        raw_response = dump_response(response)
        headers = extract_headers(response)
        final_output = getattr(response, "output_text", raw_response)

        conversation: list[Message] = []
        if self.instructions is not None:
            conversation.append(Message(role="system", content=self.instructions))
        conversation.append(Message(role="user", content=request_input))
        conversation.append(Message(role="assistant", content=final_output))

        artifacts = {
            "provider_request_id": getattr(response, "id", None),
            "raw_response": raw_response,
            "response_headers": headers or {},
        }
        rate_limit = extract_rate_limit(headers)
        if rate_limit is not None:
            artifacts["rate_limit"] = rate_limit

        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=final_output,
            conversation=conversation,
            token_usage=extract_token_usage(getattr(response, "usage", None)),
            artifacts=artifacts,
        )

    def _build_client(self) -> _OpenAIResponsesClient:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI adapter requires the optional 'openai' dependency or an injected client."
            ) from exc

        if self.base_url is not None and self.api_key is not None:
            return cast(_OpenAIResponsesClient, AsyncOpenAI(base_url=self.base_url, api_key=self.api_key))
        if self.base_url is not None:
            return cast(_OpenAIResponsesClient, AsyncOpenAI(base_url=self.base_url))
        if self.api_key is not None:
            return cast(_OpenAIResponsesClient, AsyncOpenAI(api_key=self.api_key))
        return cast(_OpenAIResponsesClient, AsyncOpenAI())


def openai(model_id: str, **kwargs: Any) -> OpenAIGenerator:
    return OpenAIGenerator(model_id, **kwargs)
