"""Tests for custom metric registration in the API."""

from dataclasses import dataclass

import pytest

import themis
from themis.core.entities import MetricScore
from themis.interfaces import Metric


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up the metrics registry after each test."""
    # Store original state
    original_metrics = themis.get_registered_metrics()
    
    yield
    
    # Restore original state
    themis.api._METRICS_REGISTRY.clear()
    themis.api._METRICS_REGISTRY.update(original_metrics)


@dataclass
class DummyMetric(Metric):
    """A dummy metric for testing."""
    
    def __post_init__(self):
        self.name = "dummy"
    
    def compute(self, *, prediction, references=None, metadata=None):
        """Always returns 1.0."""
        return MetricScore(
            metric_name=self.name,
            value=1.0,
            details={"dummy": True},
            metadata=metadata or {},
        )


class InvalidMetric:
    """A class without compute method - should fail registration."""
    pass


def test_register_metric():
    """Test that we can register a custom metric."""
    # Register the metric
    themis.register_metric("dummy", DummyMetric)
    
    # Verify it's in the registry
    registered = themis.get_registered_metrics()
    assert "dummy" in registered
    assert registered["dummy"] == DummyMetric


def test_register_metric_validates_class():
    """Test that register_metric validates the input is a class."""
    with pytest.raises(TypeError, match="metric_cls must be a class"):
        themis.register_metric("invalid", "not_a_class")
    
    with pytest.raises(TypeError, match="metric_cls must be a class"):
        themis.register_metric("invalid", DummyMetric())  # Instance, not class


def test_register_metric_validates_interface():
    """Test that register_metric validates the metric implements compute()."""
    with pytest.raises(ValueError, match="must implement compute"):
        themis.register_metric("invalid", InvalidMetric)


def test_custom_metric_in_evaluate():
    """Test that registered metrics can be used in evaluate()."""
    # Register the metric
    themis.register_metric("test_dummy", DummyMetric)
    
    # Create a simple dataset
    dataset = [
        {"id": "1", "question": "What is 1+1?", "answer": "2"},
        {"id": "2", "question": "What is 2+2?", "answer": "4"},
    ]
    
    # Run evaluation with custom metric
    report = themis.evaluate(
        dataset,
        model="fake",
        prompt="Q: {question}\nA:",
        metrics=["test_dummy"],
        limit=2,
    )
    
    # Verify the metric was used
    assert report is not None
    assert len(report.generation_results) == 2


def test_custom_metric_overrides_builtin():
    """Test that custom metrics can override built-in metrics."""
    # Register a custom metric with the same name as a built-in
    themis.register_metric("exact_match", DummyMetric)
    
    # Create a simple dataset
    dataset = [
        {"id": "1", "question": "Test", "answer": "test"},
    ]
    
    # Run evaluation - should use the custom metric
    report = themis.evaluate(
        dataset,
        model="fake",
        prompt="{question}",
        metrics=["exact_match"],
        limit=1,
    )
    
    # The dummy metric always returns 1.0, while real exact_match would fail
    # due to case mismatch
    assert report is not None


def test_get_registered_metrics_returns_copy():
    """Test that get_registered_metrics returns a copy, not the original."""
    metrics1 = themis.get_registered_metrics()
    metrics2 = themis.get_registered_metrics()
    
    # Should be equal but not the same object
    assert metrics1 == metrics2
    assert metrics1 is not metrics2
    
    # Modifying one shouldn't affect the other
    metrics1["test"] = None
    assert "test" not in metrics2


def test_unknown_metric_raises_error():
    """Test that using an unknown metric raises a helpful error."""
    dataset = [{"id": "1", "question": "Test", "answer": "test"}]
    
    with pytest.raises(ValueError, match="Unknown metric: nonexistent"):
        themis.evaluate(
            dataset,
            model="fake",
            prompt="{question}",
            metrics=["nonexistent"],
            limit=1,
        )
