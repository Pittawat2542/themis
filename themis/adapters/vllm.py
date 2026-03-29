"""vLLM OpenAI-compatible generator adapter."""

from __future__ import annotations

from typing import Any

from themis.adapters._utils import dump_response, extract_headers, extract_token_usage, stable_fingerprint
from themis.core.models import Case, GenerationResult


class VLLMGenerator:
    component_id = "generator/vllm"
    version = "1.0"

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        client: object | None = None,
        api_key: str = "EMPTY",
        api_mode: str = "responses",
        input_builder=None,
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

    async def generate(self, case: Case, ctx) -> GenerationResult:
        client = self._client or self._build_client()
        request_input = self.input_builder(case) if self.input_builder is not None else case.input
        if self.api_mode == "chat_completions":
            response = await client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": request_input}],
            )
            raw_response = dump_response(response)
            content = response.choices[0].message.content
            usage = extract_token_usage(getattr(response, "usage", None))
        else:
            response = await client.responses.create(model=self.model_id, input=request_input)
            raw_response = dump_response(response)
            content = getattr(response, "output_text", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))

        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=content,
            token_usage=usage,
            artifacts={
                "provider_request_id": getattr(response, "id", None),
                "raw_response": raw_response,
                "response_headers": extract_headers(response) or {},
                "api_mode": self.api_mode,
            },
        )

    def _build_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "vLLM adapter requires the optional 'openai' dependency or an injected client."
            ) from exc
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)


def vllm(model_id: str, **kwargs) -> VLLMGenerator:
    return VLLMGenerator(model_id, **kwargs)
