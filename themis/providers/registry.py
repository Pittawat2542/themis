"""Simple registry for ModelProvider factories."""

from __future__ import annotations

from typing import Callable, Dict

from themis.interfaces import ModelProvider

ProviderFactory = Callable[..., ModelProvider]


class _ProviderRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.lower()
        self._factories[key] = factory

    def create(self, name: str, **options) -> ModelProvider:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"No provider registered under name '{name}'")
        return factory(**options)


_REGISTRY = _ProviderRegistry()


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY.register(name, factory)


def create_provider(name: str, **options) -> ModelProvider:
    return _REGISTRY.create(name, **options)


__all__ = ["register_provider", "create_provider", "ProviderFactory"]
