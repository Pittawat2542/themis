"""Simple test for metadata propagation - no complex fixtures."""

import themis
from themis.core.entities import MetricScore
from themis.interfaces import Metric


def test_metadata_includes_all_dataset_fields():
    """Test that all dataset fields are available in metric metadata."""

    # Clean registry
    themis.api._METRICS_REGISTRY.clear()

    # Simple metric that stores what it receives
    class SimpleMetric(Metric):
        def __init__(self):
            self.name = "simple"
            self.last_metadata = None

        def compute(self, *, prediction, references=None, metadata=None):
            self.last_metadata = metadata
            return MetricScore(
                metric_name=self.name,
                value=1.0,
                details={},
                metadata=metadata or {},
            )

    metric = SimpleMetric()
    themis.api._METRICS_REGISTRY["simple"] = lambda: metric

    # Dataset with various field types
    dataset = [
        {
            "id": "test1",
            "question": "Q1",
            "answer": "A1",
            # Custom fields that metrics might need
            "numbers": [1, 2, 3],
            "target": 6,
            "category": "math",
            "difficulty": 5,
            "nested": {"key": "value"},
        }
    ]

    themis.evaluate(
        dataset,
        model="fake",
        prompt="Q: {question}\nA:",
        metrics=["simple"],
    )

    # Check what the metric received
    assert metric.last_metadata is not None
    metadata = metric.last_metadata

    # All custom fields should be present
    assert "numbers" in metadata, f"Missing 'numbers' in {metadata.keys()}"
    assert metadata["numbers"] == [1, 2, 3]

    assert "target" in metadata
    assert metadata["target"] == 6

    assert "category" in metadata
    assert metadata["category"] == "math"

    assert "difficulty" in metadata
    assert metadata["difficulty"] == 5

    assert "nested" in metadata
    assert metadata["nested"]["key"] == "value"

    # Standard fields should also be present
    assert "sample_id" in metadata
    assert "dataset_id" in metadata

    # Question field (not id or answer) should be preserved
    assert "question" in metadata
    assert metadata["question"] == "Q1"

    # Answer and id should NOT be in metadata (they're used for other purposes)
    # Note: 'id' might be present as 'dataset_id'


if __name__ == "__main__":
    test_metadata_includes_all_dataset_fields()
    print("✓ Test passed!")
