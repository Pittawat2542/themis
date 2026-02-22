from __future__ import annotations

from themis.core import entities as core_entities
from themis.generation.router import ProviderRouter
from themis.interfaces import StatelessTaskExecutor
from tests.factories import make_task


class ProviderA(StatelessTaskExecutor):
    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="A"),
            error=None,
        )


class ProviderB(StatelessTaskExecutor):
    def execute(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="B"),
            error=None,
        )


def _make_task(provider: str):
    return make_task(sample_id=provider, provider=provider, model_id="model-x")


def test_provider_router_dispatches_by_provider_and_model():
    router = ProviderRouter(
        {
            ("p1", "model-x"): ProviderA(),
            ("p2", "model-x"): ProviderB(),
        }
    )

    record_a = router.execute(_make_task("p1"))
    record_b = router.execute(_make_task("p2"))

    assert record_a.output.text == "A"
    assert record_b.output.text == "B"


def test_provider_router_falls_back_to_identifier():
    router = ProviderRouter({"model-x": ProviderA()})

    task = _make_task("model-x")
    # Identifier-only keys should be addressed directly.
    record = router.execute(task)

    assert record.output.text == "A"


def test_provider_router_requires_provider_prefix():
    router = ProviderRouter({("p1", "model-x"): ProviderA()})

    task = _make_task("p2")

    try:
        router.execute(task)
        assert False, "Expected missing provider mapping"
    except RuntimeError as exc:
        assert "No provider registered" in str(exc)
