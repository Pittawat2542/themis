"""Tests for summary export functionality."""

import json
from pathlib import Path

import pytest

from themis.core import entities as core_entities
from themis.evaluation.reports import EvaluationReport, MetricAggregate
from themis.experiment.export import export_summary_json
from themis.experiment.orchestrator import ExperimentReport


@pytest.fixture
def sample_experiment_report():
    """Create a sample experiment report."""
    # Create generation records
    gen_records = []
    for i in range(10):
        task = core_entities.GenerationTask(
            prompt=core_entities.PromptRender(
                spec=core_entities.PromptSpec(
                    name="test-prompt",
                    template="Question: {q}",
                ),
                text=f"Question: {i}",
            ),
            model=core_entities.ModelSpec(
                identifier="test-model",
                provider="test",
            ),
            sampling=core_entities.SamplingConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=100,
            ),
            metadata={"sample_id": f"sample-{i}"},
        )
        record = core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=f"Answer {i}"),
            error=None,
            metrics={"cost_usd": 0.01},
        )
        gen_records.append(record)

    # Create evaluation records
    eval_records = []
    metric_scores = []
    for i in range(10):
        score = core_entities.MetricScore(
            metric_name="accuracy",
            value=1.0 if i < 7 else 0.0,
            metadata={"sample_id": f"sample-{i}"},
        )
        metric_scores.append(score)
        eval_records.append(
            core_entities.EvaluationRecord(
                sample_id=f"sample-{i}",
                scores=[score],
            )
        )

    # Create evaluation report
    eval_report = EvaluationReport(
        metrics={
            "accuracy": MetricAggregate.from_scores("accuracy", metric_scores)
        },
        failures=[],
        records=eval_records,
    )

    return ExperimentReport(
        generation_results=gen_records,
        evaluation_report=eval_report,
        failures=[],
        metadata={"experiment": "test"},
    )


def test_summary_export_basic(tmp_path, sample_experiment_report):
    """Test basic summary export."""
    output_path = tmp_path / "summary.json"
    
    result_path = export_summary_json(
        sample_experiment_report,
        output_path,
        run_id="test-run-123",
    )
    
    assert result_path == output_path
    assert output_path.exists()
    
    # Load and verify structure
    with output_path.open("r") as f:
        summary = json.load(f)
    
    assert summary["run_id"] == "test-run-123"
    assert summary["total_samples"] == 10
    assert "metrics" in summary
    assert "metadata" in summary
    assert "cost_usd" in summary
    assert "failures" in summary


def test_summary_contains_metrics(tmp_path, sample_experiment_report):
    """Test that summary contains metric aggregates."""
    output_path = tmp_path / "summary.json"
    
    export_summary_json(
        sample_experiment_report,
        output_path,
        run_id="test-run",
    )
    
    with output_path.open("r") as f:
        summary = json.load(f)
    
    assert "accuracy" in summary["metrics"]
    accuracy = summary["metrics"]["accuracy"]
    assert "mean" in accuracy
    assert "count" in accuracy
    assert accuracy["mean"] == 0.7  # 7 out of 10
    assert accuracy["count"] == 10


def test_summary_contains_metadata(tmp_path, sample_experiment_report):
    """Test that summary contains model metadata."""
    output_path = tmp_path / "summary.json"
    
    export_summary_json(
        sample_experiment_report,
        output_path,
        run_id="test-run",
    )
    
    with output_path.open("r") as f:
        summary = json.load(f)
    
    metadata = summary["metadata"]
    assert metadata["model"] == "test-model"
    assert metadata["prompt_template"] == "test-prompt"
    assert "sampling" in metadata
    assert metadata["sampling"]["temperature"] == 0.7
    assert metadata["sampling"]["top_p"] == 0.9
    assert metadata["sampling"]["max_tokens"] == 100


def test_summary_contains_cost(tmp_path, sample_experiment_report):
    """Test that summary contains total cost."""
    output_path = tmp_path / "summary.json"
    
    export_summary_json(
        sample_experiment_report,
        output_path,
        run_id="test-run",
    )
    
    with output_path.open("r") as f:
        summary = json.load(f)
    
    # 10 samples * 0.01 each = 0.1
    assert summary["cost_usd"] == 0.1


def test_summary_file_size(tmp_path, sample_experiment_report):
    """Test that summary file is small."""
    output_path = tmp_path / "summary.json"
    
    export_summary_json(
        sample_experiment_report,
        output_path,
        run_id="test-run",
    )
    
    file_size = output_path.stat().st_size
    # Summary should be < 2KB (typically ~1KB)
    assert file_size < 2000, f"Summary file too large: {file_size} bytes"


def test_summary_with_failures(tmp_path):
    """Test summary with evaluation failures."""
    # Create report with failures
    gen_records = [
        core_entities.GenerationRecord(
            task=core_entities.GenerationTask(
                prompt=core_entities.PromptRender(
                    spec=core_entities.PromptSpec(name="test", template="Q"),
                    text="Q",
                ),
                model=core_entities.ModelSpec(identifier="test", provider="test"),
                sampling=core_entities.SamplingConfig(
                    temperature=0.7, top_p=0.9, max_tokens=100
                ),
            ),
            output=core_entities.ModelOutput(text="A"),
            error=None,
        )
    ]
    
    eval_report = EvaluationReport(
        metrics={},
        failures=[
            {"sample_id": "1", "message": "Error 1"},
            {"sample_id": "2", "message": "Error 2"},
        ],
        records=[],
    )
    
    report = ExperimentReport(
        generation_results=gen_records,
        evaluation_report=eval_report,
        failures=[],
        metadata={},
    )
    
    output_path = tmp_path / "summary.json"
    export_summary_json(report, output_path, run_id="test")
    
    with output_path.open("r") as f:
        summary = json.load(f)
    
    assert summary["failures"] == 2
    assert summary["failure_rate"] == 2.0  # 2 failures / 1 sample


def test_summary_without_run_id(tmp_path, sample_experiment_report):
    """Test summary export without run_id."""
    output_path = tmp_path / "summary.json"
    
    export_summary_json(
        sample_experiment_report,
        output_path,
    )
    
    with output_path.open("r") as f:
        summary = json.load(f)
    
    assert summary["run_id"] is None
