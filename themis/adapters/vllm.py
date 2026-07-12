"""vLLM OpenAI-compatible generator adapter."""

from __future__ import annotations

from typing import Any, Protocol, cast

from themis.adapters._utils import (
    call_maybe_sync,
    close_maybe_sync,
    dump_response,
    extract_provider_telemetry,
    extract_token_usage,
    provider_artifacts,
    stable_fingerprint,
)
from themis.core.contexts import GenerationContext
from themis.core.models import (
    Candidate,
    Case,
    Message,
    GenerationTurn,
    SeedCapability,
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
    seed_capability = SeedCapability.SUPPORTED

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        client: _VLLMClient | None = None,
        api_key: str = "EMPTY",
        api_mode: str = "responses",
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self.api_key = api_key
        self.api_mode = api_mode
        self.provider_key = f"vllm:{self.base_url}"

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "model_id": self.model_id,
                "base_url": self.base_url,
                "api_mode": self.api_mode,
                "seed_capability": self.seed_capability.value,
            }
        )

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        client = self._client_for_call()
        request_input = (
            ctx.prompt_spec.render_input(case.input)
            if ctx.prompt_spec is not None
            else case.input
        )
        if self.api_mode == "chat_completions":
            response = await call_maybe_sync(
                client.chat.completions.create,
                model=self.model_id,
                messages=[{"role": "user", "content": request_input}],
                seed=ctx.seed,
            )
            raw_response = dump_response(response)
            choices = getattr(response, "choices")
            content = getattr(choices[0].message, "content", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))
        else:
            response = await call_maybe_sync(
                client.responses.create,
                model=self.model_id, input=request_input, seed=ctx.seed
            )
            raw_response = dump_response(response)
            content = getattr(response, "output_text", raw_response)
            usage = extract_token_usage(getattr(response, "usage", None))

        telemetry = extract_provider_telemetry(
            response,
            seed_requested=ctx.seed,
            seed_applied=ctx.seed,
            seed_capability=self.seed_capability,
        )
        artifacts = provider_artifacts(telemetry)
        artifacts["api_mode"] = self.api_mode

        return Candidate(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=content,
            turns=[
                GenerationTurn(
                    turn_index=0,
                    input_messages=[Message(role="user", content=request_input)],
                    output_messages=[Message(role="assistant", content=content)],
                )
            ],
            termination_reason="completed",
            token_usage=usage or telemetry.token_usage,
            artifacts=artifacts,
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

    def _client_for_call(self) -> _VLLMClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            client, self._client = self._client, None
            await close_maybe_sync(client)

    async def __aenter__(self) -> VLLMGenerator:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def vllm(model_id: str, **kwargs: Any) -> VLLMGenerator:
    """Construct a `VLLMGenerator`."""

    return VLLMGenerator(model_id, **kwargs)
