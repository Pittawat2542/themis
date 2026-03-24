"""Provider normalization and judge spec helpers for catalog runtime."""

from __future__ import annotations

import os

from themis import InferenceParamsSpec, ModelSpec
from themis.specs.foundational import JudgeInferenceSpec
from themis.types.json_types import JSONDict


def _build_judge_spec(*, model_id: str, provider: str):
    return JudgeInferenceSpec(
        model=ModelSpec(
            model_id=model_id,
            provider=provider,
            extras=_provider_model_extras(provider),
        ),
        params=InferenceParamsSpec(max_tokens=8192, temperature=0.0),
    )


def _normalize_provider_name(provider: str) -> str:
    return provider.replace("-", "_")


def _provider_model_extras(provider: str) -> JSONDict:
    normalized = _normalize_provider_name(provider)
    if normalized == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return {"base_url": base_url.rstrip("/"), "timeout_seconds": 60.0}
    return {}
