from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from themis.config.schema import HuggingFaceHubConfig, WandbConfig
from themis.experiment.orchestrator import ExperimentReport
from themis.integrations.huggingface import HuggingFaceHubUploader
from themis.integrations.wandb import WandbTracker


@patch("wandb.init")
@patch("wandb.summary")
@patch("wandb.Table")
@patch("wandb.log")
def test_wandb_tracker(mock_log, mock_table, mock_summary, mock_init):
    config = WandbConfig(enable=True, project="test-project", entity="test-entity")
    tracker = WandbTracker(config)

    experiment_config = {"max_samples": 10, "run_id": "test-run"}
    tracker.init(experiment_config)
    mock_init.assert_called_once_with(
        project="test-project",
        entity="test-entity",
        tags=[],
        config=experiment_config,
    )

    report = MagicMock(spec=ExperimentReport)
    report.evaluation_report = MagicMock()
    report.metadata = {
        "total_samples": 10,
        "successful_generations": 8,
        "failed_generations": 2,
        "evaluation_failures": 1,
    }
    report.evaluation_report.metrics = {
        "accuracy": MagicMock(mean=0.8),
    }
    report.generation_results = []
    report.evaluation_report.records = []

    tracker.log_results(report)

    mock_summary.update.assert_called_once_with(
        {
            "total_samples": 10,
            "successful_generations": 8,
            "failed_generations": 2,
            "evaluation_failures": 1,
            "accuracy_mean": 0.8,
        }
    )
    mock_table.assert_called_once_with(
        columns=[
            "sample_id",
            "prompt",
            "raw_response",
            "parsed_response",
            "error",
            "metric_scores",
        ]
    )
    mock_log.assert_called_once()


@patch("huggingface_hub.HfApi.upload_file")
def test_huggingface_hub_uploader(mock_upload_file):
    config = HuggingFaceHubConfig(enable=True, repository="test-repo")
    uploader = HuggingFaceHubUploader(config)

    report = {
        "metadata": {"run_id": "test-run"},
        "generation_results": [],
        "evaluation_report": {"metrics": {}, "failures": [], "records": []},
        "failures": [],
    }

    storage_path = Path(".cache/test-storage")
    storage_path.mkdir(exist_ok=True, parents=True)

    # Create a dummy ExperimentReport object for the uploader
    from themis.core.entities import ExperimentReport
    from themis.evaluation.pipeline import EvaluationReport

    exp_report = ExperimentReport(
        metadata=report["metadata"],
        generation_results=report["generation_results"],
        evaluation_report=EvaluationReport(
            metrics={},
            failures=[],
            records=[],
        ),
        failures=report["failures"],
    )

    uploader.upload_results(exp_report, storage_path)

    mock_upload_file.assert_called()
