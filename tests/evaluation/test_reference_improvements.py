"""Tests for reference handling improvements."""

import warnings

import pytest

from themis.core import entities as core_entities
from themis.evaluation import extractors
from themis.evaluation.pipelines.standard_pipeline import (
    EvaluationPipeline,
    _normalize_references,
)
from themis.interfaces import Metric


class SimpleExtractor:
    """Simple extractor for testing."""

    def extract(self, raw_output: str):
        return raw_output.strip()


class TestMetric(Metric):
    """Test metric that records what it receives."""

    name = "test_metric"

    def __init__(self):
        self.last_prediction = None
        self.last_references = None
        self.last_metadata = None

    def compute(self, *, prediction, references, metadata=None):
        self.last_prediction = prediction
        self.last_references = references
        self.last_metadata = metadata
        return core_entities.MetricScore(
            metric_name=self.name,
            value=1.0,
        )


def test_normalize_references_with_reference_object():
    """Test reference normalization with Reference object."""
    ref = core_entities.Reference(kind="answer", value="42")
    normalized = _normalize_references(ref)
    assert normalized == ["42"]


def test_normalize_references_with_dict_value():
    """Test reference normalization with dict value in Reference."""
    ref = core_entities.Reference(
        kind="task",
        value={"target": 122, "numbers": [25, 50, 75, 100]},
    )
    normalized = _normalize_references(ref)
    assert len(normalized) == 1
    assert normalized[0] == {"target": 122, "numbers": [25, 50, 75, 100]}


def test_normalize_references_with_list():
    """Test reference normalization with list."""
    refs = ["answer1", "answer2", "answer3"]
    normalized = _normalize_references(refs)
    assert normalized == ["answer1", "answer2", "answer3"]


def test_normalize_references_with_tuple():
    """Test reference normalization with tuple."""
    refs = ("answer1", "answer2")
    normalized = _normalize_references(refs)
    assert normalized == ["answer1", "answer2"]


def test_normalize_references_with_scalar():
    """Test reference normalization with scalar value."""
    normalized = _normalize_references("42")
    assert normalized == ["42"]
    
    normalized = _normalize_references(42)
    assert normalized == [42]


def test_reference_selector_precedence():
    """Test that custom reference_selector takes precedence."""
    # Create task with reference
    task = core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=core_entities.PromptSpec(name="test", template="Q"),
            text="Q",
        ),
        model=core_entities.ModelSpec(identifier="test", provider="test"),
        sampling=core_entities.SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=100
        ),
        reference=core_entities.Reference(kind="default", value="default_ref"),
        metadata={"custom_ref": "custom_value"},
    )
    
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="output"),
        error=None,
    )
    
    # Custom reference selector that returns different value
    def custom_selector(rec):
        return rec.task.metadata["custom_ref"]
    
    metric = TestMetric()
    
    # Create pipeline with custom selector (should warn)
    with warnings.catch_warnings(record=True) as w:
        pipeline = EvaluationPipeline(
            extractor=SimpleExtractor(),
            metrics=[metric],
            reference_selector=custom_selector,
        )
        assert len(w) == 1
        assert "reference_selector" in str(w[0].message)
    
    # Evaluate
    pipeline.evaluate([record])
    
    # Verify custom selector was used (not default)
    assert metric.last_references == ["custom_value"]


def test_reference_selector_with_dict_return():
    """Test reference selector returning dict for multi-value reference."""
    task = core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=core_entities.PromptSpec(name="test", template="Q"),
            text="Q",
        ),
        model=core_entities.ModelSpec(identifier="test", provider="test"),
        sampling=core_entities.SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=100
        ),
        metadata={
            "target": 122,
            "numbers": [25, 50, 75, 100],
        },
    )
    
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="output"),
        error=None,
    )
    
    # Reference selector that returns dict
    def multi_value_selector(rec):
        return {
            "target": rec.task.metadata["target"],
            "numbers": rec.task.metadata["numbers"],
        }
    
    metric = TestMetric()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = EvaluationPipeline(
            extractor=SimpleExtractor(),
            metrics=[metric],
            reference_selector=multi_value_selector,
        )
    
    pipeline.evaluate([record])
    
    # Verify dict reference is preserved
    assert len(metric.last_references) == 1
    ref = metric.last_references[0]
    assert isinstance(ref, dict)
    assert ref["target"] == 122
    assert ref["numbers"] == [25, 50, 75, 100]


def test_no_warning_without_reference_selector():
    """Test no warning when not using custom reference_selector."""
    with warnings.catch_warnings(record=True) as w:
        EvaluationPipeline(
            extractor=SimpleExtractor(),
            metrics=[TestMetric()],
        )
        # Should not warn
        assert len(w) == 0


def test_dict_value_in_reference_object():
    """Test using dict value directly in Reference object."""
    ref = core_entities.Reference(
        kind="countdown_task",
        value={"target": 122, "numbers": [25, 50, 75, 100]},
    )
    
    # Access dict fields
    assert ref.value["target"] == 122
    assert ref.value["numbers"] == [25, 50, 75, 100]
    
    # Normalize for metric
    normalized = _normalize_references(ref)
    assert len(normalized) == 1
    assert normalized[0]["target"] == 122


def test_extractor_runs_before_metric():
    """Test that extractor processes output before metric receives it."""
    class AnswerExtractor:
        def extract(self, raw_output: str):
            # Extract answer from tags
            if "<answer>" in raw_output:
                start = raw_output.index("<answer>") + 8
                end = raw_output.index("</answer>")
                return raw_output[start:end]
            return raw_output
    
    task = core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=core_entities.PromptSpec(name="test", template="Q"),
            text="Q",
        ),
        model=core_entities.ModelSpec(identifier="test", provider="test"),
        sampling=core_entities.SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=100
        ),
        reference=core_entities.Reference(kind="answer", value="42"),
    )
    
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(
            text="<think>Let me calculate</think><answer>42</answer>"
        ),
        error=None,
    )
    
    metric = TestMetric()
    
    pipeline = EvaluationPipeline(
        extractor=AnswerExtractor(),
        metrics=[metric],
    )
    
    pipeline.evaluate([record])
    
    # Metric should receive extracted answer, not raw output
    assert metric.last_prediction == "42"
    assert "<answer>" not in metric.last_prediction
    assert "<think>" not in metric.last_prediction


def test_metadata_passed_to_metric():
    """Test that metadata is passed to metrics."""
    task = core_entities.GenerationTask(
        prompt=core_entities.PromptRender(
            spec=core_entities.PromptSpec(name="test", template="Q"),
            text="Q",
        ),
        model=core_entities.ModelSpec(identifier="test", provider="test"),
        sampling=core_entities.SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=100
        ),
        metadata={"dataset_id": "sample-123", "difficulty": "hard"},
        reference=core_entities.Reference(kind="answer", value="42"),
    )
    
    record = core_entities.GenerationRecord(
        task=task,
        output=core_entities.ModelOutput(text="42"),
        error=None,
    )
    
    metric = TestMetric()
    
    pipeline = EvaluationPipeline(
        extractor=SimpleExtractor(),
        metrics=[metric],
    )
    
    pipeline.evaluate([record])
    
    # Verify metadata contains sample_id
    assert metric.last_metadata is not None
    assert metric.last_metadata["sample_id"] == "sample-123"
