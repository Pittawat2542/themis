"""Utility router mapping generation tasks to providers."""

from __future__ import annotations

from collections.abc import Mapping

from themis.core import entities as core_entities
from themis.exceptions import ProviderError
from themis.interfaces import StatelessTaskExecutor


ProviderKey = str | tuple[str, str]


def _model_key(provider: str, identifier: str) -> str:
    return f"{provider}:{identifier}"


class ProviderRouter(StatelessTaskExecutor):
    """Dispatches generation tasks to concrete providers by model identifier."""

    def __init__(self, providers: Mapping[ProviderKey, StatelessTaskExecutor]):
        normalized: dict[str, StatelessTaskExecutor] = {}

        for key, provider in providers.items():
            if isinstance(key, tuple):
                provider_name, model_id = key
                normalized[_model_key(provider_name, model_id)] = provider
            else:
                normalized[str(key)] = provider

        self._providers = normalized

    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        provider = self._providers.get(
            _model_key(task.model.provider, task.model.identifier)
        )
        if provider is None:
            provider = self._providers.get(task.model.identifier)
        if provider is None:
            known = ", ".join(sorted(self._providers)) or "<none>"
            raise ProviderError(
                f"No provider registered for model '{task.model.identifier}'. "
                f"Known providers: {known}."
            )
        return provider.execute(task)

    @property
    def providers(self) -> Mapping[str, StatelessTaskExecutor]:
        return self._providers


__all__ = ["ProviderRouter"]
