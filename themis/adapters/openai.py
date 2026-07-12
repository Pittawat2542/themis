"""OpenAI Responses API generator adapter."""

from __future__ import annotations

from typing import Any, Protocol, cast

from themis.adapters._utils import (
    call_maybe_sync,
    close_maybe_sync,
    extract_provider_telemetry,
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


class _OpenAIResponsesClient(Protocol):
    @property
    def responses(self) -> _ResponsesCreateAPI: ...


class OpenAIGenerator:
    """Generator adapter for the OpenAI Responses API."""

    component_id = "generator/openai"
    version = "1.0"
    seed_capability = SeedCapability.SUPPORTED

    def __init__(
        self,
        model_id: str,
        *,
        client: _OpenAIResponsesClient | None = None,
        instructions: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client
        self._owns_client = client is None
        self.instructions = instructions
        self.base_url = base_url
        self.api_key = api_key
        self.provider_key = (
            f"openai:{(base_url or 'https://api.openai.com/v1').rstrip('/')}"
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "model_id": self.model_id,
                "instructions": self.instructions,
                "base_url": self.base_url,
                "seed_capability": self.seed_capability.value,
            }
        )

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        client = self._client_for_call()
        prompt_spec = ctx.prompt_spec
        request_input = case.input
        rendered_input = (
            prompt_spec.model_copy(update={"instructions": None}).render_input(
                request_input
            )
            if prompt_spec is not None
            else request_input
        )
        instructions = self.instructions or (
            prompt_spec.instructions if prompt_spec is not None else None
        )
        payload: dict[str, object] = {"model": self.model_id, "input": rendered_input}
        if ctx.seed is not None:
            payload["seed"] = ctx.seed
        if instructions is not None:
            payload["instructions"] = instructions

        response = await call_maybe_sync(client.responses.create, **payload)
        telemetry = extract_provider_telemetry(
            response,
            seed_requested=ctx.seed,
            seed_applied=ctx.seed,
            seed_capability=self.seed_capability,
        )
        final_output = getattr(response, "output_text", telemetry.raw_response)

        conversation: list[Message] = []
        if instructions is not None:
            conversation.append(Message(role="system", content=instructions))
        conversation.append(Message(role="user", content=rendered_input))
        conversation.append(Message(role="assistant", content=final_output))

        return Candidate(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=final_output,
            turns=[
                GenerationTurn(
                    turn_index=0,
                    input_messages=conversation[:-1],
                    output_messages=conversation[-1:],
                )
            ],
            conversation=conversation,
            termination_reason="completed",
            token_usage=telemetry.token_usage,
            artifacts=provider_artifacts(telemetry),
        )

    def _build_client(self) -> _OpenAIResponsesClient:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "OpenAI adapter requires the optional 'openai' dependency or an injected client."
            ) from exc

        if self.base_url is not None and self.api_key is not None:
            return cast(
                _OpenAIResponsesClient,
                AsyncOpenAI(base_url=self.base_url, api_key=self.api_key),
            )
        if self.base_url is not None:
            return cast(_OpenAIResponsesClient, AsyncOpenAI(base_url=self.base_url))
        if self.api_key is not None:
            return cast(_OpenAIResponsesClient, AsyncOpenAI(api_key=self.api_key))
        return cast(_OpenAIResponsesClient, AsyncOpenAI())

    def _client_for_call(self) -> _OpenAIResponsesClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            client, self._client = self._client, None
            await close_maybe_sync(client)

    async def __aenter__(self) -> OpenAIGenerator:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def openai(model_id: str, **kwargs: Any) -> OpenAIGenerator:
    """Construct an `OpenAIGenerator`."""

    return OpenAIGenerator(model_id, **kwargs)
