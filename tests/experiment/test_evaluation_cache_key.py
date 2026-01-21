"""Unit tests for evaluation cache key functionality."""

import pytest

from themis.core import entities as core_entities
from themis.experiment.storage import evaluation_cache_key, task_cache_key


def make_test_task(dataset_id: str = "test-1", temperature: float = 0.7) -> core_entities.GenerationTask:
    """Create a test generation task."""
    prompt_spec = core_entities.PromptSpec(name="math", template="Solve {problem}")
    prompt_render = core_entities.PromptRender(
        spec=prompt_spec, text="Solve 1+1", context={"problem": "1+1"}, metadata={}
    )
    sampling = core_entities.SamplingConfig(temperature=temperature, top_p=1.0, max_tokens=100)
    model_spec = core_entities.ModelSpec(identifier="gpt-4", provider="openai")
    return core_entities.GenerationTask(
        prompt=prompt_render,
        model=model_spec,
        sampling=sampling,
        metadata={"dataset_id": dataset_id},
        reference=core_entities.Reference(kind="answer", value="2"),
    )


def test_evaluation_cache_key_without_config():
    """Test that evaluation_cache_key without config returns task key."""
    task = make_test_task()
    task_key = task_cache_key(task)
    eval_key = evaluation_cache_key(task, None)
    
    assert eval_key == task_key, "Should match task key when no config provided"


def test_evaluation_cache_key_with_config():
    """Test that evaluation_cache_key with config includes eval marker."""
    task = make_test_task()
    task_key = task_cache_key(task)
    
    config = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer"
    }
    eval_key = evaluation_cache_key(task, config)
    
    assert eval_key != task_key, "Should differ from task key when config provided"
    assert "::eval:" in eval_key, "Should contain eval marker"
    assert eval_key.startswith(task_key), "Should start with task key"


def test_evaluation_cache_key_invalidates_on_metric_change():
    """Test that adding/removing metrics changes the cache key."""
    task = make_test_task()
    
    config_one_metric = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer"
    }
    config_two_metrics = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer"
    }
    
    key_one = evaluation_cache_key(task, config_one_metric)
    key_two = evaluation_cache_key(task, config_two_metrics)
    
    assert key_one != key_two, "Different metrics should produce different keys"


def test_evaluation_cache_key_invalidates_on_extractor_change():
    """Test that changing extractor changes the cache key."""
    task = make_test_task()
    
    config_answer = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer"
    }
    config_solution = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:solution"
    }
    
    key_answer = evaluation_cache_key(task, config_answer)
    key_solution = evaluation_cache_key(task, config_solution)
    
    assert key_answer != key_solution, "Different extractors should produce different keys"


def test_evaluation_cache_key_stable_for_same_config():
    """Test that same config produces same key."""
    task = make_test_task()
    
    config = {
        "metrics": ["exact_match", "f1_score"],
        "extractor": "json_field_extractor:answer"
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
        "extractor": "json_field_extractor:answer"
    }
    config_b = {
        "extractor": "json_field_extractor:answer",
        "metrics": ["exact_match", "f1_score"]
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
    assert len(parts) == 5, "Task key should have 5 parts"
    # dataset_id::template::model::sampling::prompt_hash
    
    assert parts[0] == "test-1", "First part should be dataset_id"
    assert parts[1] == "math", "Second part should be template"
    assert parts[2] == "gpt-4", "Third part should be model"
    assert "0.700-1.000-100" in parts[3], "Fourth part should be sampling"


def test_evaluation_cache_key_format():
    """Test evaluation cache key format."""
    task = make_test_task()
    config = {
        "metrics": ["exact_match"],
        "extractor": "json_field_extractor:answer"
    }
    
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
    """Test that empty config dict behaves like None."""
    task = make_test_task()
    task_key = task_cache_key(task)
    
    eval_key_none = evaluation_cache_key(task, None)
    eval_key_empty = evaluation_cache_key(task, {})
    
    # Empty config is falsy in Python (empty dict), so should match task key
    assert eval_key_none == task_key, "None config should return task key"
    assert eval_key_empty == task_key, "Empty dict is falsy, should return task key"
