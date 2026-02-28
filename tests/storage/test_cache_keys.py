"""Tests for cache key generation extracted from storage/core.py."""

from themis.core import entities as core_entities


class TestCacheKeyGeneration:
    """Verify cache key functions work identically after extraction."""

    def test_task_cache_key_deterministic(self):
        from themis.storage.cache_keys import task_cache_key

        task = self._make_task()
        key1 = task_cache_key(task)
        key2 = task_cache_key(task)
        assert key1 == key2
        assert isinstance(key1, str)

    def test_evaluation_cache_key_varies_with_config(self):
        from themis.storage.cache_keys import evaluation_cache_key

        task = self._make_task()
        key1 = evaluation_cache_key(task, {"metrics": ["ExactMatch"]})
        key2 = evaluation_cache_key(task, {"metrics": ["BLEU"]})
        assert key1 != key2

    def test_backward_compat_import_from_storage(self):
        """Existing import path must still work."""
        from themis.storage import task_cache_key, evaluation_cache_key

        assert callable(task_cache_key)
        assert callable(evaluation_cache_key)

    def _make_task(self):
        # Shared test fixture
        sampling = core_entities.SamplingConfig(0.1, 0.9, 64)
        prompt_spec = core_entities.PromptSpec(name="test", template="{q}")
        prompt = core_entities.PromptRender(
            spec=prompt_spec, text="hello", context={"q": "hi"}
        )
        model = core_entities.ModelSpec(identifier="gpt-4", provider="test")
        return core_entities.GenerationTask(
            prompt=prompt, model=model, sampling=sampling, metadata={}
        )
