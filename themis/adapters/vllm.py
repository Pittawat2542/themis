"""vLLM OpenAI-compatible generator adapter."""

from __future__ import annotations

from typing import Any, Protocol, cast

from themis.adapters._utils import (
    dump_response,
    extract_provider_telemetry,
    extract_token_usage,
    provider_artifacts,
    stable_fingerprint,
)
from themis.core.contexts import GenerateContext, SessionContext
from themis.core.models import (
    Case,
    GenerationResult,
    Message,
    SessionResult,
    SessionTurn,
)


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

    async def run_session(self, case: Case, ctx: SessionContext) -> SessionResult:
        client = self._client or self._build_client()
        request_input = (
            self.input_builder(case) if self.input_builder is not None else case.input
        )
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
            response = await client.responses.create(
                model=self.model_id, input=request_input
            )
            raw_response = dump_response(response)
            content = getattr(response, "output_text", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))

        telemetry = extract_provider_telemetry(response)
        artifacts = provider_artifacts(telemetry)
        artifacts["api_mode"] = self.api_mode

        return SessionResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=content,
            turns=[
                SessionTurn(
                    turn_index=0,
                    input_messages=[Message(role="user", content=request_input)],
                    output_messages=[Message(role="assistant", content=content)],
                )
            ],
            termination_reason="completed",
            token_usage=usage or telemetry.token_usage,
            artifacts=artifacts,
        )

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        return GenerationResult.model_validate(
            (await self.run_session(case, ctx)).model_dump(mode="json")
        )

    def _build_client(self) -> _VLLMClient:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "vLLM adapter requires the Linux-only 'vllm' extra or an injected OpenAI-compatible client."
            ) from exc
        return cast(
            _VLLMClient, AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        )


def vllm(model_id: str, **kwargs: Any) -> VLLMGenerator:
    """Construct a `VLLMGenerator`."""

    return VLLMGenerator(model_id, **kwargs)
