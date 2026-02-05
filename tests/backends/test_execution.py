"""Tests for execution backends."""

import time

import pytest

from themis.backends.execution import (
    ExecutionBackend,
    LocalExecutionBackend,
    SequentialExecutionBackend,
)
from themis.core import entities as core_entities
from themis.generation.runner import GenerationRunner
from themis.interfaces import ModelProvider


class TestLocalExecutionBackend:
    """Tests for LocalExecutionBackend."""
    
    def test_map_basic(self):
        """Test basic map functionality."""
        backend = LocalExecutionBackend(max_workers=2)
        
        items = [1, 2, 3, 4, 5]
        results = list(backend.map(lambda x: x * 2, items))
        
        assert len(results) == 5
        assert set(results) == {2, 4, 6, 8, 10}
        
        backend.shutdown()
    
    def test_map_with_workers_override(self):
        """Test overriding max_workers in map."""
        backend = LocalExecutionBackend(max_workers=2)
        
        items = [1, 2, 3]
        results = list(backend.map(lambda x: x * 2, items, max_workers=4))
        
        assert len(results) == 3
        backend.shutdown()
    
    def test_parallel_execution(self):
        """Test that execution is actually parallel."""
        backend = LocalExecutionBackend(max_workers=3)
        
        def slow_func(x):
            time.sleep(0.1)
            return x * 2
        
        items = [1, 2, 3]
        
        start = time.time()
        results = list(backend.map(slow_func, items))
        duration = time.time() - start
        
        # With 3 workers, should take ~0.1s, not 0.3s
        assert duration < 0.25  # Some overhead allowed
        assert len(results) == 3
        
        backend.shutdown()
    
    def test_context_manager(self):
        """Test using backend as context manager."""
        with LocalExecutionBackend(max_workers=2) as backend:
            results = list(backend.map(lambda x: x * 2, [1, 2, 3]))
            assert len(results) == 3
        
        # After exit, executor should be shutdown
        assert backend._executor is None or backend._executor._shutdown
    
    def test_error_handling(self):
        """Test error handling in parallel execution."""
        backend = LocalExecutionBackend(max_workers=2)
        
        def failing_func(x):
            if x == 2:
                raise ValueError("Test error")
            return x * 2
        
        with pytest.raises(ValueError, match="Test error"):
            list(backend.map(failing_func, [1, 2, 3]))
        
        backend.shutdown()


class TestSequentialExecutionBackend:
    """Tests for SequentialExecutionBackend."""
    
    def test_map_basic(self):
        """Test basic sequential map."""
        backend = SequentialExecutionBackend()
        
        items = [1, 2, 3, 4, 5]
        results = list(backend.map(lambda x: x * 2, items))
        
        assert results == [2, 4, 6, 8, 10]  # Order preserved
        backend.shutdown()
    
    def test_sequential_order(self):
        """Test that results are in input order."""
        backend = SequentialExecutionBackend()
        
        call_order = []
        
        def track_func(x):
            call_order.append(x)
            return x * 2
        
        items = [3, 1, 4, 1, 5]
        results = list(backend.map(track_func, items))
        
        assert call_order == [3, 1, 4, 1, 5]
        assert results == [6, 2, 8, 2, 10]
        
        backend.shutdown()
    
    def test_no_parallelism(self):
        """Test that execution is truly sequential."""
        backend = SequentialExecutionBackend()
        
        execution_times = []
        
        def timed_func(x):
            start = time.time()
            time.sleep(0.05)
            execution_times.append(time.time() - start)
            return x
        
        start = time.time()
        list(backend.map(timed_func, [1, 2, 3]))
        total_time = time.time() - start
        
        # Should take ~0.15s (3 * 0.05s)
        assert total_time >= 0.14  # At least sequential
        
        backend.shutdown()
    
    def test_error_handling(self):
        """Test error handling in sequential execution."""
        backend = SequentialExecutionBackend()
        
        def failing_func(x):
            if x == 2:
                raise ValueError("Test error")
            return x * 2
        
        results = []
        with pytest.raises(ValueError, match="Test error"):
            for result in backend.map(failing_func, [1, 2, 3]):
                results.append(result)
        
        assert len(results) == 1  # Only first item processed before error
        backend.shutdown()


class TestExecutionBackendInterface:
    """Tests for ExecutionBackend interface."""
    
    def test_abstract_methods(self):
        """Test that ExecutionBackend cannot be instantiated."""
        with pytest.raises(TypeError):
            ExecutionBackend()
    
    def test_context_manager_protocol(self):
        """Test that backends support context manager protocol."""
        backend = SequentialExecutionBackend()
        
        assert hasattr(backend, '__enter__')
        assert hasattr(backend, '__exit__')
        
        with backend as b:
            assert b is backend


class RecordingExecutionBackend(ExecutionBackend):
    def __init__(self):
        self.called = False
        self.max_workers = None
        self.shutdown_called = False

    def map(self, func, items, *, max_workers=None, timeout=None, **kwargs):
        self.called = True
        self.max_workers = max_workers
        for item in items:
            yield func(item)

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeProvider(ModelProvider):
    def generate(self, task: core_entities.GenerationTask) -> core_entities.GenerationRecord:
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text="ok"),
            error=None,
        )


def _make_task(sample_id: str) -> core_entities.GenerationTask:
    prompt_spec = core_entities.PromptSpec(name="t", template="Q")
    prompt = core_entities.PromptRender(spec=prompt_spec, text="Q")
    model = core_entities.ModelSpec(identifier="model-x", provider="fake")
    sampling = core_entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=8)
    return core_entities.GenerationTask(
        prompt=prompt,
        model=model,
        sampling=sampling,
        metadata={"dataset_id": sample_id},
    )


def test_generation_runner_uses_execution_backend():
    backend = RecordingExecutionBackend()
    runner = GenerationRunner(provider=FakeProvider(), execution_backend=backend, max_parallel=2)
    tasks = [_make_task("a"), _make_task("b")]

    results = list(runner.run(tasks))

    assert backend.called is True
    assert backend.max_workers == 2
    assert backend.shutdown_called is False
    assert len(results) == 2
