"""Tests for the provider registry."""

import pytest

from themis.core import entities as core_entities
from themis.exceptions import ProviderError
from themis.providers import create_provider, list_providers, register_provider
from themis.interfaces import StatelessTaskExecutor


class DummyProvider(StatelessTaskExecutor):
    def __init__(self):
        self.calls: list[core_entities.GenerationTask] = []

    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        self.calls.append(task)
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="dummy"),
            error=None,
            metrics={},
        )


def test_provider_registry_returns_registered_provider():
    register_provider("dummy-test", DummyProvider)
    provider = create_provider("dummy-test")

    prompt_spec = core_entities.PromptSpec(name="t", template="{x}")
    prompt = core_entities.PromptRender(
        spec=prompt_spec, text="", context={}, metadata={}
    )
    model = core_entities.ModelSpec(identifier="dummy", provider="dummy-test")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    task = core_entities.GenerationTask(prompt=prompt, model=model, sampling=sampling)

    record = provider.execute(task)

    assert record.output.text == "dummy"


def test_provider_registry_unknown_provider_raises():
    """ProviderError is raised with available providers listed."""
    with pytest.raises(ProviderError, match="Available providers"):
        create_provider("non-existent-provider-xyz")


def test_provider_registry_unknown_is_also_key_error():
    """ProviderError inherits from KeyError for backward compat."""
    with pytest.raises(KeyError):
        create_provider("non-existent-provider-xyz")


def test_list_providers_returns_sorted_list():
    register_provider("zeta-test", DummyProvider)
    register_provider("alpha-test", DummyProvider)
    providers = list_providers()
    assert isinstance(providers, list)
    assert "alpha-test" in providers
    assert "zeta-test" in providers
    # Should be sorted
    assert providers == sorted(providers)
