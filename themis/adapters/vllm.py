"""vLLM OpenAI-compatible generator adapter."""

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
from themis.core.models import Case, GenerationResult


class _ResponsesCreateAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _ChatCompletionsAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _ChatClient(Protocol):
    @property
    def completions(self) -> _ChatCompletionsAPI: ...


class _VLLMClient(Protocol):
    @property
    def responses(self) -> _ResponsesCreateAPI: ...

    @property
    def chat(self) -> _ChatClient: ...


class VLLMGenerator:
    """Generator adapter for vLLM's OpenAI-compatible endpoints."""

    component_id = "generator/vllm"
    version = "1.0"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        client: _VLLMClient | None = None,
        api_key: str = "EMPTY",
        api_mode: str = "responses",
        input_builder: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.api_key = api_key
        self.api_mode = api_mode
        self.input_builder = input_builder
        self.provider_key = f"vllm:{self.base_url}"

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "model_id": self.model_id,
                "base_url": self.base_url,
                "api_mode": self.api_mode,
            }
        )

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        client = self._client or self._build_client()
        request_input = self.input_builder(case) if self.input_builder is not None else case.input
        if self.api_mode == "chat_completions":
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": request_input}],
            )
            raw_response = dump_response(response)
            choices = getattr(response, "choices")
            content = getattr(choices[0].message, "content", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))
        else:
            response = await client.responses.create(model=self.model_id, input=request_input)
            raw_response = dump_response(response)
            content = getattr(response, "output_text", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))

        headers = extract_headers(response)
        artifacts = {
            "provider_request_id": getattr(response, "id", None),
            "raw_response": raw_response,
            "response_headers": headers or {},
            "api_mode": self.api_mode,
        }
        rate_limit = extract_rate_limit(headers)
        if rate_limit is not None:
            artifacts["rate_limit"] = rate_limit

        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=content,
            token_usage=usage,
            artifacts=artifacts,
        )

    def _build_client(self) -> _VLLMClient:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "vLLM adapter requires the Linux-only 'vllm' extra or an injected OpenAI-compatible client."
            ) from exc
        return cast(_VLLMClient, AsyncOpenAI(base_url=self.base_url, api_key=self.api_key))


def vllm(model_id: str, **kwargs: Any) -> VLLMGenerator:
    """Construct a `VLLMGenerator`."""

    return VLLMGenerator(model_id, **kwargs)
