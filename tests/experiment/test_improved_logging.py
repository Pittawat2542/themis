"""Tests for improved logging in math experiment."""

from themis.core import entities as core_entities
from themis.evaluation import pipeline as evaluation_pipeline
from themis.experiment import math as math_experiment, orchestrator


def test_improved_summarize_report_without_math_verify():
    """Test the improved summarize_report function without MathVerify."""
    # Create a mock evaluation report
    exact_match_metric = evaluation_pipeline.MetricAggregate(
        name="ExactMatch",
        count=8,
        mean=0.75,
        per_sample=[
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 0.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 0.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 0.0, {}, {}),
        ],
    )

    eval_report = evaluation_pipeline.EvaluationReport(
        metrics={"ExactMatch": exact_match_metric},
        failures=[],  # No evaluation failures
        records=[],
    )

    # Create experiment report with metadata
    report = orchestrator.ExperimentReport(
        generation_results=[],
        evaluation_report=eval_report,
        failures=[],  # No generation failures
        metadata={
            "total_samples": 10,
            "successful_generations": 8,
            "failed_generations": 2,
        },
    )

    # Test the summarize function
    summary = math_experiment.summarize_report(report)

    # Check that the summary contains the expected information
    assert "Evaluated 10 samples" in summary
    assert "Successful generations: 8/10" in summary
    assert "Exact match: 0.750 (8 evaluated)" in summary
    assert "No failures" in summary


def test_improved_summarize_report_with_failures():
    """Test the improved summarize_report function with failures."""
    # Create a mock evaluation report
    exact_match_metric = evaluation_pipeline.MetricAggregate(
        name="ExactMatch",
        count=5,
        mean=0.6,
        per_sample=[
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 0.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 0.0, {}, {}),
            core_entities.MetricScore("ExactMatch", 1.0, {}, {}),
        ],
    )

    eval_report = evaluation_pipeline.EvaluationReport(
        metrics={"ExactMatch": exact_match_metric},
        failures=[
            evaluation_pipeline.EvaluationFailure("sample-1", "Extraction failed"),
            evaluation_pipeline.EvaluationFailure(
                "sample-2", "Metric computation failed"
            ),
        ],
        records=[],
    )

    # Create experiment report with metadata
    report = orchestrator.ExperimentReport(
        generation_results=[],
        evaluation_report=eval_report,
        failures=[
            orchestrator.ExperimentFailure("sample-3", "Generation timeout"),
            orchestrator.ExperimentFailure("sample-4", "API error"),
        ],
        metadata={
            "total_samples": 10,
            "successful_generations": 5,
            "failed_generations": 5,
        },
    )

    # Test the summarize function
    summary = math_experiment.summarize_report(report)

    # Check that the summary contains the expected information
    assert "Evaluated 10 samples" in summary
    assert "Successful generations: 5/10" in summary
    assert "Exact match: 0.600 (5 evaluated)" in summary
    assert "Failures: 4 (gen: 5, eval: 2)" in summary


def test_improved_summarize_report_with_math_verify():
    """Test the improved summarize_report function with MathVerify."""
    # Create a mock evaluation report with both ExactMatch and MathVerifyAccuracy
    exact_match_metric = evaluation_pipeline.MetricAggregate(
        name="ExactMatch",
        count=10,
        mean=0.7,
        per_sample=[],
    )

    math_verify_metric = evaluation_pipeline.MetricAggregate(
        name="MathVerifyAccuracy",
        count=10,
        mean=0.8,
        per_sample=[],
    )

    eval_report = evaluation_pipeline.EvaluationReport(
        metrics={
            "ExactMatch": exact_match_metric,
            "MathVerifyAccuracy": math_verify_metric,
        },
        failures=[],
        records=[],
    )

    # Create experiment report with metadata
    report = orchestrator.ExperimentReport(
        generation_results=[],
        evaluation_report=eval_report,
        failures=[],
        metadata={
            "total_samples": 10,
            "successful_generations": 10,
            "failed_generations": 0,
        },
    )

    # Test the summarize function
    summary = math_experiment.summarize_report(report)

    # Check that the summary contains the expected information
    assert "Evaluated 10 samples" in summary
    assert "Successful generations: 10/10" in summary
    assert "Exact match: 0.700 (10 evaluated)" in summary
    assert "MathVerify accuracy: 0.800 (10 evaluated)" in summary
    assert "No failures" in summary
