"""Simple registry for model provider factories."""

from __future__ import annotations

from typing import Any, Callable

from themis.exceptions import ProviderError
from themis.interfaces import StatelessTaskExecutor
from themis.presets import parse_model_name
from themis.providers.options import normalize_provider_options

ProviderFactory = Callable[..., StatelessTaskExecutor]


class _ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        key = name.lower()
        self._factories[key] = factory

    def create(self, name: str, **options) -> StatelessTaskExecutor:
        key = name.lower()
        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys())) or "(none)"
            raise ProviderError(
                f"No provider registered under name '{name}'. "
                f"Available providers: {available}. "
                f"Register a custom provider with themis.register_provider()."
            )
        return factory(**options)

    def list_providers(self) -> list[str]:
        return sorted(self._factories.keys())

    def _reset_for_testing(self) -> None:
        """Reset to empty state. For testing only."""
        self._factories.clear()


_REGISTRY = _ProviderRegistry()


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a model provider factory in the global registry.

    Registered providers can be used by passing their name as a model
    prefix in ``themis.evaluate()`` (e.g., ``model="myprovider:model-id"``).

    Args:
        name: Provider identifier (case-insensitive).
        factory: Callable that returns a ``StatelessTaskExecutor`` instance.

    Example:
        >>> from themis import register_provider
        >>> from themis.interfaces import StatelessTaskExecutor
        >>>
        >>> class MyProvider(StatelessTaskExecutor):
        ...     def execute(self, task):
        ...         ...
        >>>
        >>> register_provider("myprovider", MyProvider)
    """
    _REGISTRY.register(name, factory)


def create_provider(name: str, **options) -> StatelessTaskExecutor:
    """Create a provider instance from the registry.

    Args:
        name: Registered provider identifier.
        **options: Provider-specific options (e.g., ``api_key``).

    Returns:
        A ``StatelessTaskExecutor`` ready for generation.

    Raises:
        ProviderError: If the provider name is not registered.
    """
    return _REGISTRY.create(name, **options)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        Sorted list of registered provider identifiers.

    Example:
        >>> import themis
        >>> themis.list_providers()
        ['fake', 'litellm', 'vllm']
    """
    return _REGISTRY.list_providers()


def _reset_for_testing() -> None:
    """Reset provider registry. For testing only."""
    _REGISTRY._reset_for_testing()


def parse_model(model: str, **provider_options: Any) -> tuple[str, str, dict[str, Any]]:
    """Parse model string into provider, model_id, and options."""
    options = normalize_provider_options(provider_options)
    if ":" in model:
        provider_name, model_id = model.split(":", 1)
        return provider_name, model_id, options

    parsed_provider, model_id, parsed_options = parse_model_name(model, **options)
    return parsed_provider, model_id, normalize_provider_options(parsed_options)


__all__ = [
    "register_provider",
    "create_provider",
    "list_providers",
    "parse_model",
    "ProviderFactory",
]
