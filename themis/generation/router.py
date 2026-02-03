"""Utility router mapping generation tasks to providers."""

from __future__ import annotations

from typing import Mapping, Tuple, Union

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider


ProviderKey = Union[str, Tuple[str, str]]


def _model_key(provider: str, identifier: str) -> str:
    return f"{provider}:{identifier}"


class ProviderRouter(ModelProvider):
    """Dispatches generation tasks to concrete providers by model identifier."""

    def __init__(self, providers: Mapping[ProviderKey, ModelProvider]):
        normalized: dict[str, ModelProvider] = {}
        identifier_counts: dict[str, int] = {}

        for key, provider in providers.items():
            if isinstance(key, tuple):
                provider_name, model_id = key
                normalized[_model_key(provider_name, model_id)] = provider
                identifier_counts[model_id] = identifier_counts.get(model_id, 0) + 1
            else:
                normalized[str(key)] = provider
                identifier_counts[str(key)] = identifier_counts.get(str(key), 0) + 1

        # Add identifier-only aliases when they are unambiguous
        for key, provider in list(normalized.items()):
            if ":" in key:
                _, model_id = key.split(":", 1)
                if identifier_counts.get(model_id, 0) == 1 and model_id not in normalized:
                    normalized[model_id] = provider

        self._providers = normalized

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        provider = self._providers.get(_model_key(task.model.provider, task.model.identifier))
        if provider is None:
            provider = self._providers.get(task.model.identifier)
        if provider is None:
            known = ", ".join(sorted(self._providers)) or "<none>"
            raise RuntimeError(
                f"No provider registered for model '{task.model.identifier}'. "
                f"Known providers: {known}."
            )
        return provider.generate(task)

    @property
    def providers(self) -> Mapping[str, ModelProvider]:
        return self._providers


__all__ = ["ProviderRouter"]
