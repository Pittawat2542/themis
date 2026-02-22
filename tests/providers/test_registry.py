import pytest

from themis.core import entities as core_entities
from themis.providers import create_provider, register_provider
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
    with pytest.raises(KeyError):
        create_provider("non-existent-provider")
