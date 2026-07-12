"""Provider adapters with shared telemetry extraction."""

from __future__ import annotations

from typing import Any

from themis.adapters._utils import (
    call_maybe_sync,
    extract_provider_telemetry,
    normalize_json_value,
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
from themis.core.workflows import JudgeResponse


class ProviderAdapter:
    version = "1.0"
    seed_capability = SeedCapability.UNSUPPORTED

    def __init__(
        self,
        provider: str,
        model_id: str,
        *,
        client: object | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.component_id = f"generator/{provider}"
        self._client = client
        self.endpoint = endpoint
        self.base_url = base_url
        self.provider_key = (
            f"{provider}:{(endpoint or base_url or model_id).rstrip('/')}"
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "provider": self.provider,
                "model_id": self.model_id,
                "endpoint": self.endpoint,
                "base_url": self.base_url,
                "seed_capability": self.seed_capability.value,
            }
        )

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        request_input = _render_input(case, ctx)
        response = await self._invoke(str(request_input))
        telemetry = extract_provider_telemetry(
            response,
            seed_requested=ctx.seed,
            seed_capability=self.seed_capability,
        )
        final_output = _extract_text(response)
        conversation = [
            Message(role="user", content=normalize_json_value(request_input)),
            Message(role="assistant", content=final_output),
        ]
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

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        response = await self._invoke(prompt)
        telemetry = extract_provider_telemetry(
            response,
            seed_requested=seed,
            seed_capability=self.seed_capability,
        )
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            effective_seed=None,
            raw_response=_extract_text(response),
            token_usage=telemetry.token_usage or {},
            provider_request_id=telemetry.request_id,
        )

    async def _invoke(self, prompt: str) -> object:
        client = self._require_client()
        if self.provider == "anthropic":
            return await call_maybe_sync(
                client.messages.create,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
        if self.provider == "bedrock":
            return await call_maybe_sync(
                client.converse,
                modelId=self.model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]},
                ],
            )
        if self.provider == "gemini":
            return await call_maybe_sync(
                client.models.generate_content, model=self.model_id, contents=prompt
            )
        if self.provider == "azure_openai":
            return await call_maybe_sync(
                client.responses.create, model=self.model_id, input=prompt
            )
        if self.provider == "ollama":
            return await call_maybe_sync(
                client.generate, model=self.model_id, prompt=prompt
            )
        if self.provider == "litellm":
            if hasattr(client, "acompletion"):
                return await call_maybe_sync(
                    client.acompletion,
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                )
            return await call_maybe_sync(
                client.completion,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _require_client(self) -> Any:
        if self._client is None:
            raise ImportError(
                f"{self.provider} adapter requires an injected client or provider SDK."
            )
        return self._client


def anthropic(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("anthropic", model_id, **kwargs)


def bedrock(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("bedrock", model_id, **kwargs)


def gemini(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("gemini", model_id, **kwargs)


def azure_openai(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("azure_openai", model_id, **kwargs)


def ollama(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("ollama", model_id, **kwargs)


def litellm(model_id: str, **kwargs: Any) -> ProviderAdapter:
    return ProviderAdapter("litellm", model_id, **kwargs)


def _render_input(case: Case, ctx: GenerationContext) -> object:
    if ctx.prompt_spec is None:
        return case.input
    return ctx.prompt_spec.render_input(case.input)


def _extract_text(response: object) -> str:
    for attr in ("output_text", "text", "response"):
        value = getattr(response, attr, None)
        if value is not None:
            return str(value)
    if isinstance(response, dict):
        extracted = _extract_text_from_mapping(response)
        if extracted is not None:
            return extracted
    content = getattr(response, "content", None)
    extracted = _extract_text_from_content(content)
    if extracted is not None:
        return extracted
    choices = getattr(response, "choices", None)
    extracted = _extract_text_from_choices(choices)
    if extracted is not None:
        return extracted
    return str(normalize_json_value(response))


def _extract_text_from_mapping(response: dict[object, object]) -> str | None:
    for key in ("output_text", "text", "response"):
        value = response.get(key)
        if value is not None:
            return str(value)
    choices = response.get("choices")
    extracted = _extract_text_from_choices(choices)
    if extracted is not None:
        return extracted
    output = response.get("output")
    if isinstance(output, dict):
        message = output.get("message")
        if isinstance(message, dict):
            return _extract_text_from_content(message.get("content"))
    return _extract_text_from_content(response.get("content"))


def _extract_text_from_content(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                pieces.append(str(item["text"]))
            elif getattr(item, "text", None) is not None:
                pieces.append(str(getattr(item, "text")))
        return "".join(pieces) if pieces else None
    return None


def _extract_text_from_choices(choices: object) -> str | None:
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    return str(content) if content is not None else None
