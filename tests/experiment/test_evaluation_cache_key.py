"""Unit tests for evaluation cache key functionality."""

import hashlib


from themis.core import entities as core_entities
from themis.storage import (
    ExperimentStorage,
    evaluation_cache_key,
    task_cache_key,
)


def make_test_task(
    dataset_id: str = "test-1", temperature: float = 0.7
) -> core_entities.GenerationTask:
    """Create a test generation task."""
    prompt_spec = core_entities.PromptSpec(name="math", template="Solve {problem}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Solve 1+1", context={"problem": "1+1"}, metadata={}
    )
    sampling = core_entities.SamplingConfig(
        temperature=temperature, top_p=1.0, max_tokens=100
    )
    model_spec = core_entities.ModelSpec(identifier="gpt-4", provider="openai")
    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": dataset_id},
        reference=core_entities.Reference(kind="answer", value="2"),
    )


def test_evaluation_cache_key_without_config():
    """Test that evaluation_cache_key without config still includes eval fingerprint."""
    task = make_test_task()
    task_key = task_cache_key(task)
    eval_key = evaluation_cache_key(task, None)

    assert eval_key.startswith(task_key), "Should keep task key prefix"
    assert "::eval:" in eval_key, "Should include canonical eval fingerprint"


def test_evaluation_cache_key_with_config():
    """Test that evaluation_cache_key with config includes eval marker."""
    task = make_test_task()
    task_key = task_cache_key(task)

    config = {"metrics": ["exact_match"], "extractor": "json_field_extractor:answer"}
    eval_key = evaluation_cache_key(task, config)

    assert eval_key != task_key, "Should differ from task key when config provided"
    assert "::eval:" in eval_key, "Should contain eval marker"
    assert eval_key.startswith(task_key), "Should start with task key"


def test_evaluation_cache_key_invalidates_on_metric_change():
    """Test that adding/removing metrics changes the cache key."""
    task = make_test_task()

    config_one_metric = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer",
    }
    config_two_metrics = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer",
    }

    key_one = evaluation_cache_key(task, config_one_metric)
    key_two = evaluation_cache_key(task, config_two_metrics)

    assert key_one != key_two, "Different metrics should produce different keys"


def test_evaluation_cache_key_invalidates_on_extractor_change():
    """Test that changing extractor changes the cache key."""
    task = make_test_task()

    config_answer = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer",
    }
    config_solution = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:solution",
    }

    key_answer = evaluation_cache_key(task, config_answer)
    key_solution = evaluation_cache_key(task, config_solution)

    assert key_answer != key_solution, (
        "Different extractors should produce different keys"
    )


def test_evaluation_cache_key_stable_for_same_config():
    """Test that same config produces same key."""
    task = make_test_task()

    config = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer",
    }

    key1 = evaluation_cache_key(task, config)
    key2 = evaluation_cache_key(task, config)

    assert key1 == key2, "Same config should produce same key"


def test_evaluation_cache_key_handles_unsorted_config():
    """Test that config order doesn't matter (deterministic)."""
    task = make_test_task()

    # Same config, different order
    config_a = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer",
    }
    config_b = {
        "extractor": "json_field_extractor:answer",
        "metrics": ["exact_match", "f1_score"],
    }

    key_a = evaluation_cache_key(task, config_a)
    key_b = evaluation_cache_key(task, config_b)

    assert key_a == key_b, "Config order should not affect key (deterministic)"


def test_task_cache_key_includes_sampling():
    """Test that task cache key includes sampling parameters."""
    task1 = make_test_task(temperature=0.7)
    task2 = make_test_task(temperature=0.9)

    key1 = task_cache_key(task1)
    key2 = task_cache_key(task2)

    assert key1 != key2, "Different temperatures should produce different task keys"
    assert "0.700" in key1, "Task key should contain temperature"
    assert "0.900" in key2, "Task key should contain temperature"


def test_task_cache_key_format():
    """Test task cache key format."""
    task = make_test_task()
    key = task_cache_key(task)

    parts = key.split("::")
    assert len(parts) == 7, "Task key should have version + 6 data components"
    # k2::dataset_id::template::model::sampling::prompt_hash::reference_hash

    assert parts[0] == "k2", "First part should be cache key version"
    assert parts[1] == "test-1", "Second part should be dataset_id"
    assert parts[2] == "math", "Third part should be template"
    assert parts[3] == "openai:gpt-4", "Fourth part should be model"
    assert "0.700-1.000-100" in parts[4], "Fifth part should be sampling"
    assert len(parts[5]) == 12, "Prompt hash should be 12 chars"
    assert len(parts[6]) == 12, "Reference hash should be 12 chars"


def test_task_cache_key_includes_manifest_hash():
    """Test that task cache key includes manifest hash when provided."""
    task = make_test_task()
    task.metadata["manifest_hash"] = "deadbeef"

    key = task_cache_key(task)

    assert key.endswith("deadbeef"), "Task key should include manifest hash suffix"


def test_task_cache_key_invalidates_on_manifest_hash_change():
    """Test that different manifest hashes produce different task cache keys."""
    task_a = make_test_task()
    task_b = make_test_task()
    task_a.metadata["manifest_hash"] = "aaa"
    task_b.metadata["manifest_hash"] = "bbb"

    key_a = task_cache_key(task_a)
    key_b = task_cache_key(task_b)

    assert key_a != key_b


def test_evaluation_cache_key_format():
    """Test evaluation cache key format."""
    task = make_test_task()
    config = {"metrics": ["exact_match"], "extractor": "json_field_extractor:answer"}

    eval_key = evaluation_cache_key(task, config)
    task_key = task_cache_key(task)

    # Should be: {task_key}::eval:{hash}
    assert eval_key.startswith(task_key), "Should start with task key"
    assert "::eval:" in eval_key, "Should have eval separator"

    # Extract hash part
    eval_part = eval_key.split("::eval:")
    assert len(eval_part) == 2, "Should have exactly one eval separator"
    assert len(eval_part[1]) == 12, "Hash should be 12 characters"


def test_evaluation_cache_key_with_empty_config():
    """Test that omitted/empty config use the same canonical default hash."""
    task = make_test_task()
    task_key = task_cache_key(task)

    eval_key_none = evaluation_cache_key(task, None)
    eval_key_empty = evaluation_cache_key(task, {})

    assert eval_key_none.startswith(task_key)
    assert eval_key_none == eval_key_empty


def test_task_cache_key_invalidates_on_reference_change():
    task_a = make_test_task()
    task_b = make_test_task()
    task_b.reference = core_entities.Reference(kind="answer", value="3")

    assert task_cache_key(task_a) != task_cache_key(task_b)


def test_new_cache_keys_do_not_false_hit_legacy_records(tmp_path):
    storage = ExperimentStorage(tmp_path)
    run_id = "legacy-run"
    storage.start_run(run_id, experiment_id="default")

    task = make_test_task()
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="2"),
        error=None,
        metrics={},
    )
    old_key = _legacy_task_cache_key(task)
    storage.append_record(run_id, record, cache_key=old_key)

    loaded = storage.load_cached_records(run_id)
    assert old_key in loaded  # old data remains readable
    assert task_cache_key(task) not in loaded  # no false hit under new key format


def _legacy_task_cache_key(task: core_entities.GenerationTask) -> str:
    dataset_raw = task.metadata.get("dataset_id") or task.metadata.get("sample_id")
    dataset_id = str(dataset_raw) if dataset_raw is not None else ""
    prompt_hash = hashlib.sha256(task.prompt.text.encode("utf-8")).hexdigest()[:12]
    sampling = task.sampling
    sampling_key = (
        f"{sampling.temperature:.3f}-{sampling.top_p:.3f}-{sampling.max_tokens}"
    )
    return "::".join(
        [
            dataset_id,
            task.prompt.spec.name,
            task.model.identifier,
            sampling_key,
            prompt_hash,
        ]
    )
