"""Tests for batch optimization system."""

import pytest

from themis.core import entities
from themis.generation import batching, templates


def create_test_tasks(count: int = 10) -> list[entities.GenerationTask]:
    """Create test generation tasks."""
    tasks = []

    models = [
        entities.ModelSpec(identifier="model-a", provider="fake"),
        entities.ModelSpec(identifier="model-b", provider="fake"),
    ]

    samplings = [
        entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100),
        entities.SamplingConfig(temperature=0.7, top_p=0.95, max_tokens=100),
    ]

    template = templates.PromptTemplate(name="test", template="Test prompt {i}")

    for i in range(count):
        model = models[i % len(models)]
        sampling = samplings[i % len(samplings)]
        prompt = template.render_prompt({"i": i})

        task = entities.GenerationTask(
            prompt=prompt,
            model=model,
            sampling=sampling,
            metadata={"task_id": i, "group": i % 3},
        )
        tasks.append(task)

    return tasks


def test_batch_config_validation():
    """Test BatchConfig validates parameters."""
    # Valid config
    config = batching.BatchConfig(max_batch_size=10)
    assert config.max_batch_size == 10

    # Invalid batch size
    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        batching.BatchConfig(max_batch_size=0)

    with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
        batching.BatchConfig(max_batch_size=-5)

    # Invalid timeout
    with pytest.raises(ValueError, match="timeout_ms must be >= 0"):
        batching.BatchConfig(max_batch_size=10, timeout_ms=-1)


def test_task_batcher_simple_batching():
    """Test simple batching without grouping."""
    tasks = create_test_tasks(25)

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # 25 tasks with max_batch_size=10 should create 3 batches
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_task_batcher_exact_batch_size():
    """Test batching when task count is exact multiple of batch size."""
    tasks = create_test_tasks(20)

    config = batching.BatchConfig(max_batch_size=5)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # 20 tasks with max_batch_size=5 should create exactly 4 batches
    assert len(batches) == 4
    assert all(len(batch) == 5 for batch in batches)


def test_task_batcher_single_batch():
    """Test batching when all tasks fit in one batch."""
    tasks = create_test_tasks(5)

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_task_batcher_empty_tasks():
    """Test batching with empty task list."""
    tasks = []

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    assert len(batches) == 0


def test_task_batcher_group_by_model():
    """Test batching with model grouping."""
    tasks = create_test_tasks(20)  # 10 model-a, 10 model-b

    config = batching.BatchConfig(
        max_batch_size=5,
        group_by=batching.group_by_model,
    )
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # Should create 4 batches (2 per model)
    assert len(batches) == 4

    # Each batch should contain tasks from same model
    for batch in batches:
        models = {task.model.identifier for task in batch}
        assert len(models) == 1  # All same model


def test_task_batcher_group_by_custom():
    """Test batching with custom grouping function."""
    tasks = create_test_tasks(18)  # 3 groups of 6

    # Group by metadata "group" field
    def group_by_metadata(task):
        return f"group_{task.metadata.get('group')}"

    config = batching.BatchConfig(
        max_batch_size=5,
        group_by=group_by_metadata,
    )
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # 18 tasks in 3 groups, batch_size=5
    # Group 0: 6 tasks -> 2 batches (5, 1)
    # Group 1: 6 tasks -> 2 batches (5, 1)
    # Group 2: 6 tasks -> 2 batches (5, 1)
    # Total: 6 batches
    assert len(batches) == 6

    # Verify each batch has tasks from same group
    for batch in batches:
        groups = {task.metadata.get("group") for task in batch}
        assert len(groups) == 1


def test_group_by_model():
    """Test group_by_model helper function."""
    model_a = entities.ModelSpec(identifier="model-a", provider="fake")
    model_b = entities.ModelSpec(identifier="model-b", provider="fake")

    template = templates.PromptTemplate(name="test", template="Test")
    prompt = template.render_prompt({})
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    task_a = entities.GenerationTask(prompt=prompt, model=model_a, sampling=sampling)
    task_b = entities.GenerationTask(prompt=prompt, model=model_b, sampling=sampling)

    assert batching.group_by_model(task_a) == "model-a"
    assert batching.group_by_model(task_b) == "model-b"


def test_group_by_prompt_length():
    """Test group_by_prompt_length helper function."""
    model = entities.ModelSpec(identifier="model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Short prompt (50 chars)
    short_template = templates.PromptTemplate(name="short", template="a" * 50)
    short_prompt = short_template.render_prompt({})
    short_task = entities.GenerationTask(
        prompt=short_prompt, model=model, sampling=sampling
    )

    # Long prompt (250 chars)
    long_template = templates.PromptTemplate(name="long", template="a" * 250)
    long_prompt = long_template.render_prompt({})
    long_task = entities.GenerationTask(
        prompt=long_prompt, model=model, sampling=sampling
    )

    # Default bucket size is 100
    assert batching.group_by_prompt_length(short_task) == "length_0-100"
    assert batching.group_by_prompt_length(long_task) == "length_200-300"

    # Custom bucket size
    assert (
        batching.group_by_prompt_length(short_task, bucket_size=50) == "length_50-100"
    )
    assert (
        batching.group_by_prompt_length(long_task, bucket_size=50) == "length_250-300"
    )


def test_group_by_model_and_sampling():
    """Test group_by_model_and_sampling helper function."""
    model_a = entities.ModelSpec(identifier="model-a", provider="fake")
    model_b = entities.ModelSpec(identifier="model-b", provider="fake")

    sampling_1 = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)
    sampling_2 = entities.SamplingConfig(temperature=0.7, top_p=0.8, max_tokens=100)

    template = templates.PromptTemplate(name="test", template="Test")
    prompt = template.render_prompt({})

    task_a1 = entities.GenerationTask(prompt=prompt, model=model_a, sampling=sampling_1)
    task_a2 = entities.GenerationTask(prompt=prompt, model=model_a, sampling=sampling_2)
    task_b1 = entities.GenerationTask(prompt=prompt, model=model_b, sampling=sampling_1)

    key_a1 = batching.group_by_model_and_sampling(task_a1)
    key_a2 = batching.group_by_model_and_sampling(task_a2)
    key_b1 = batching.group_by_model_and_sampling(task_b1)

    # Same model + sampling should have same key
    assert key_a1 == "model-a_t0.0_p0.95"

    # Different sampling should have different key
    assert key_a2 == "model-a_t0.7_p0.8"

    # Different model should have different key
    assert key_b1 == "model-b_t0.0_p0.95"

    # All should be different
    assert len({key_a1, key_a2, key_b1}) == 3


def test_create_grouping_function():
    """Test composite grouping function creator."""
    model_a = entities.ModelSpec(identifier="model-a", provider="fake")
    model_b = entities.ModelSpec(identifier="model-b", provider="fake")

    # Short and long templates
    short_template = templates.PromptTemplate(name="short", template="a" * 50)
    long_template = templates.PromptTemplate(name="long", template="a" * 250)

    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    # Create composite grouper
    composite_grouper = batching.create_grouping_function(
        batching.group_by_model,
        batching.group_by_prompt_length,
    )

    # Tasks with different models and lengths
    task_a_short = entities.GenerationTask(
        prompt=short_template.render_prompt({}),
        model=model_a,
        sampling=sampling,
    )

    task_a_long = entities.GenerationTask(
        prompt=long_template.render_prompt({}),
        model=model_a,
        sampling=sampling,
    )

    task_b_short = entities.GenerationTask(
        prompt=short_template.render_prompt({}),
        model=model_b,
        sampling=sampling,
    )

    # Get composite keys
    key_a_short = composite_grouper(task_a_short)
    key_a_long = composite_grouper(task_a_long)
    key_b_short = composite_grouper(task_b_short)

    # Keys should combine both groupers
    assert "model-a" in key_a_short
    assert "length_0-100" in key_a_short

    assert "model-a" in key_a_long
    assert "length_200-300" in key_a_long

    assert "model-b" in key_b_short
    assert "length_0-100" in key_b_short

    # All should be different
    assert len({key_a_short, key_a_long, key_b_short}) == 3


def test_get_batch_count():
    """Test get_batch_count helper method."""
    tasks = create_test_tasks(25)

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    count = batcher.get_batch_count(tasks)
    assert count == 3


def test_get_batch_stats():
    """Test get_batch_stats helper method."""
    tasks = create_test_tasks(25)

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    stats = batcher.get_batch_stats(tasks)

    assert stats["total_tasks"] == 25
    assert stats["num_batches"] == 3
    assert stats["max_batch_size"] == 10
    assert stats["min_batch_size"] == 5
    assert stats["avg_batch_size"] == pytest.approx(8.333, rel=0.01)


def test_get_batch_stats_with_grouping():
    """Test get_batch_stats includes group information."""
    tasks = create_test_tasks(20)

    config = batching.BatchConfig(
        max_batch_size=5,
        group_by=batching.group_by_model,
    )
    batcher = batching.TaskBatcher(config)

    stats = batcher.get_batch_stats(tasks)

    assert stats["total_tasks"] == 20
    assert stats["num_batches"] == 4
    assert stats["num_groups"] == 2
    assert "group_sizes" in stats
    assert len(stats["group_sizes"]) == 2


def test_batch_order_preserved():
    """Test that batching preserves order within batches."""
    tasks = create_test_tasks(15)

    config = batching.BatchConfig(max_batch_size=5)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # Reconstruct task order from batches
    reconstructed = []
    for batch in batches:
        reconstructed.extend(batch)

    # Should be in same order
    original_ids = [t.metadata["task_id"] for t in tasks]
    reconstructed_ids = [t.metadata["task_id"] for t in reconstructed]
    assert original_ids == reconstructed_ids


def test_batch_with_single_task():
    """Test batching with a single task."""
    tasks = create_test_tasks(1)

    config = batching.BatchConfig(max_batch_size=10)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_all_tasks_different_models():
    """Test batching when every task has different model."""
    # Create tasks with unique models
    tasks = []
    for i in range(10):
        model = entities.ModelSpec(identifier=f"model-{i}", provider="fake")
        template = templates.PromptTemplate(name="test", template="Test")
        prompt = template.render_prompt({})
        sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

        task = entities.GenerationTask(
            prompt=prompt,
            model=model,
            sampling=sampling,
            metadata={"task_id": i},
        )
        tasks.append(task)

    # Group by model with batch size 5
    config = batching.BatchConfig(
        max_batch_size=5,
        group_by=batching.group_by_model,
    )
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # Each model should get its own batch (size 1)
    assert len(batches) == 10
    assert all(len(batch) == 1 for batch in batches)


def test_batch_all_tasks_same_model():
    """Test batching when all tasks have same model."""
    model = entities.ModelSpec(identifier="same-model", provider="fake")
    template = templates.PromptTemplate(name="test", template="Test")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=0.95, max_tokens=100)

    tasks = []
    for i in range(15):
        prompt = template.render_prompt({"i": i})
        task = entities.GenerationTask(
            prompt=prompt,
            model=model,
            sampling=sampling,
            metadata={"task_id": i},
        )
        tasks.append(task)

    # Group by model with batch size 5
    config = batching.BatchConfig(
        max_batch_size=5,
        group_by=batching.group_by_model,
    )
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # All tasks in same model group: 15 / 5 = 3 batches
    assert len(batches) == 3
    assert all(len(batch) == 5 for batch in batches)


def test_batching_with_large_batch_size():
    """Test batching when batch size exceeds task count."""
    tasks = create_test_tasks(5)

    config = batching.BatchConfig(max_batch_size=100)
    batcher = batching.TaskBatcher(config)

    batches = list(batcher.create_batches(tasks))

    # All tasks should fit in one batch
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_batching_idempotent():
    """Test that batching is idempotent (same input produces same output)."""
    tasks = create_test_tasks(20)

    config = batching.BatchConfig(max_batch_size=7)
    batcher = batching.TaskBatcher(config)

    # Run batching twice
    batches1 = list(batcher.create_batches(tasks))
    batches2 = list(batcher.create_batches(tasks))

    # Should produce same batches
    assert len(batches1) == len(batches2)

    for b1, b2 in zip(batches1, batches2):
        assert len(b1) == len(b2)
        ids1 = [t.metadata["task_id"] for t in b1]
        ids2 = [t.metadata["task_id"] for t in b2]
        assert ids1 == ids2
