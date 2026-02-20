"""Tests for metadata preservation in evaluation strategies."""

from themis.core.entities import (
    GenerationRecord,
    GenerationTask,
    MetricScore,
    ModelOutput,
)
from themis.evaluation.strategies import (
    AttemptAwareEvaluationStrategy,
    JudgeEvaluationStrategy,
)


def test_attempt_aware_strategy_preserves_metadata():
    """Test that AttemptAwareEvaluationStrategy preserves original metadata."""

    # Create scores with custom metadata
    scores = [
        MetricScore(
            metric_name="accuracy",
            value=0.8,
            metadata={
                "sample_id": "test1",
                "custom_field": "value1",
                "numbers": [1, 2, 3],
            },
            details={},
        ),
        MetricScore(
            metric_name="accuracy",
            value=0.9,
            metadata={
                "sample_id": "test1",
                "custom_field": "value1",
                "numbers": [1, 2, 3],
            },
            details={},
        ),
    ]

    # Create a dummy record (not used in aggregation but required by API)
    from themis.core.entities import ModelSpec, SamplingConfig, PromptRender, PromptSpec

    dummy_task = GenerationTask(
        prompt=PromptRender(spec=PromptSpec(name="test", template="test"), text="test"),
        model=ModelSpec(identifier="test", provider="test"),
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=100),
        metadata={},
    )
    dummy_record = GenerationRecord(
        task=dummy_task, output=ModelOutput(text="test"), error=None
    )

    strategy = AttemptAwareEvaluationStrategy()
    aggregated = strategy.aggregate(dummy_record, scores)

    assert len(aggregated) == 1
    result = aggregated[0]

    # Check aggregation-specific fields
    assert result.metadata["attempts"] == 2
    assert result.metadata["sample_id"] == "test1"

    # Check original metadata is preserved
    assert "custom_field" in result.metadata
    assert result.metadata["custom_field"] == "value1"
    assert "numbers" in result.metadata
    assert result.metadata["numbers"] == [1, 2, 3]


def test_judge_evaluation_strategy_preserves_metadata():
    """Test that JudgeEvaluationStrategy preserves original metadata."""

    # Create judge scores with custom metadata
    scores = [
        MetricScore(
            metric_name="judge_score",
            value=1.0,
            metadata={
                "sample_id": "test1",
                "custom_field": "value1",
                "category": "math",
            },
            details={},
        ),
        MetricScore(
            metric_name="judge_score",
            value=1.0,
            metadata={
                "sample_id": "test1",
                "custom_field": "value1",
                "category": "math",
            },
            details={},
        ),
        MetricScore(
            metric_name="judge_score",
            value=0.0,
            metadata={
                "sample_id": "test1",
                "custom_field": "value1",
                "category": "math",
            },
            details={},
        ),
    ]

    # Create a dummy record
    from themis.core.entities import ModelSpec, SamplingConfig, PromptRender, PromptSpec

    dummy_task = GenerationTask(
        prompt=PromptRender(spec=PromptSpec(name="test", template="test"), text="test"),
        model=ModelSpec(identifier="test", provider="test"),
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=100),
        metadata={},
    )
    dummy_record = GenerationRecord(
        task=dummy_task, output=ModelOutput(text="test"), error=None
    )

    strategy = JudgeEvaluationStrategy()
    aggregated = strategy.aggregate(dummy_record, scores)

    assert len(aggregated) == 1
    result = aggregated[0]

    # Check aggregation-specific details
    assert result.details["judge_count"] == 3
    assert "agreement" in result.details
    assert "labels" in result.details

    # Check original metadata is preserved
    assert result.metadata["sample_id"] == "test1"
    assert "custom_field" in result.metadata
    assert result.metadata["custom_field"] == "value1"
    assert "category" in result.metadata
    assert result.metadata["category"] == "math"


def test_multiple_metrics_with_attempt_aware():
    """Test metadata preservation with multiple metrics."""

    scores = [
        # Metric 1 - attempt 1
        MetricScore(
            metric_name="metric1",
            value=0.7,
            metadata={
                "sample_id": "sample1",
                "field_a": "value_a",
                "field_b": 42,
            },
            details={},
        ),
        # Metric 1 - attempt 2
        MetricScore(
            metric_name="metric1",
            value=0.8,
            metadata={
                "sample_id": "sample1",
                "field_a": "value_a",
                "field_b": 42,
            },
            details={},
        ),
        # Metric 2 - attempt 1
        MetricScore(
            metric_name="metric2",
            value=0.5,
            metadata={
                "sample_id": "sample1",
                "field_c": [1, 2, 3],
            },
            details={},
        ),
        # Metric 2 - attempt 2
        MetricScore(
            metric_name="metric2",
            value=0.6,
            metadata={
                "sample_id": "sample1",
                "field_c": [1, 2, 3],
            },
            details={},
        ),
    ]

    # Create a dummy record (not used in aggregation but required by API)
    from themis.core.entities import ModelSpec, SamplingConfig, PromptRender, PromptSpec

    dummy_task = GenerationTask(
        prompt=PromptRender(spec=PromptSpec(name="test", template="test"), text="test"),
        model=ModelSpec(identifier="test", provider="test"),
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=100),
        metadata={},
    )
    dummy_record = GenerationRecord(
        task=dummy_task, output=ModelOutput(text="test"), error=None
    )

    strategy = AttemptAwareEvaluationStrategy()
    aggregated = strategy.aggregate(dummy_record, scores)

    assert len(aggregated) == 2

    # Find each metric
    metric1_result = next(s for s in aggregated if s.metric_name == "metric1")
    metric2_result = next(s for s in aggregated if s.metric_name == "metric2")

    # Check metric1 preserved its metadata
    assert metric1_result.metadata["attempts"] == 2
    assert "field_a" in metric1_result.metadata
    assert metric1_result.metadata["field_a"] == "value_a"
    assert "field_b" in metric1_result.metadata
    assert metric1_result.metadata["field_b"] == 42

    # Check metric2 preserved its metadata
    assert metric2_result.metadata["attempts"] == 2
    assert "field_c" in metric2_result.metadata
    assert metric2_result.metadata["field_c"] == [1, 2, 3]


def test_nested_metadata_preserved_in_strategies():
    """Test that nested metadata structures are preserved."""

    scores = [
        MetricScore(
            metric_name="test",
            value=1.0,
            metadata={
                "sample_id": "test1",
                "nested": {"level1": {"level2": "deep_value"}},
                "list_field": [{"item": 1}, {"item": 2}],
            },
            details={},
        ),
    ]

    # Create a dummy record (not used in aggregation but required by API)
    from themis.core.entities import ModelSpec, SamplingConfig, PromptRender, PromptSpec

    dummy_task = GenerationTask(
        prompt=PromptRender(spec=PromptSpec(name="test", template="test"), text="test"),
        model=ModelSpec(identifier="test", provider="test"),
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=100),
        metadata={},
    )
    dummy_record = GenerationRecord(
        task=dummy_task, output=ModelOutput(text="test"), error=None
    )

    strategy = AttemptAwareEvaluationStrategy()
    aggregated = strategy.aggregate(dummy_record, scores)

    result = aggregated[0]

    # Check nested structures preserved
    assert "nested" in result.metadata
    assert result.metadata["nested"]["level1"]["level2"] == "deep_value"
    assert "list_field" in result.metadata
    assert len(result.metadata["list_field"]) == 2
    assert result.metadata["list_field"][0]["item"] == 1
