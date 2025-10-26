"""Utility router mapping generation tasks to providers."""

from __future__ import annotations

from typing import Mapping

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider


class ProviderRouter(ModelProvider):
    """Dispatches generation tasks to concrete providers by model identifier."""

    def __init__(self, providers: Mapping[str, ModelProvider]):
        self._providers = dict(providers)

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
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
