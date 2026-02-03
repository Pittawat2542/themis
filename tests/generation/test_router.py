from __future__ import annotations

from themis.core import entities as core_entities
from themis.generation.router import ProviderRouter
from themis.interfaces import ModelProvider


class ProviderA(ModelProvider):
    def generate(self, task: core_entities.GenerationTask) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="A"),
            error=None,
        )


class ProviderB(ModelProvider):
    def generate(self, task: core_entities.GenerationTask) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="B"),
            error=None,
        )


def _make_task(provider: str) -> core_entities.GenerationTask:
    prompt_spec = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=prompt_spec, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider=provider)
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    return core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": provider},
    )


def test_provider_router_dispatches_by_provider_and_model():
    router = ProviderRouter(
        {
            ("p1", "model-x"): ProviderA(),
            ("p2", "model-x"): ProviderB(),
        }
    )

    record_a = router.generate(_make_task("p1"))
    record_b = router.generate(_make_task("p2"))

    assert record_a.output.text == "A"
    assert record_b.output.text == "B"


def test_provider_router_falls_back_to_identifier():
    router = ProviderRouter({"model-x": ProviderA()})

    task = _make_task("model-x")
    # Identifier-only keys should be addressed directly.
    record = router.generate(task)

    assert record.output.text == "A"


def test_provider_router_requires_provider_prefix():
    router = ProviderRouter({("p1", "model-x"): ProviderA()})

    task = _make_task("p2")

    try:
        router.generate(task)
        assert False, "Expected missing provider mapping"
    except RuntimeError as exc:
        assert "No provider registered" in str(exc)
