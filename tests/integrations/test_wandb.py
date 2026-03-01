from unittest.mock import patch
from themis.config.schema import WandbConfig
from themis.integrations.wandb import WandbTracker
from themis.core.entities import ExperimentReport
from themis.evaluation.reports import EvaluationReport


class MockAggregate:
    def __init__(self, mean, count):
        self.mean = mean
        self.count = count


@patch("themis.integrations.wandb.wandb")
def test_wandb_init_enabled(mock_wandb):
    config = WandbConfig(
        enable=True, project="test-project", entity="test-entity", tags=["a", "b"]
    )
    tracker = WandbTracker(config)

    tracker.init({"run_id": "123"})
    mock_wandb.init.assert_called_once_with(
        project="test-project",
        entity="test-entity",
        tags=["a", "b"],
        config={"run_id": "123"},
    )


@patch("themis.integrations.wandb.wandb")
def test_wandb_log_results(mock_wandb):
    config = WandbConfig(enable=True, project="test")
    tracker = WandbTracker(config)

    report = ExperimentReport(
        metadata={"total_samples": 5},
        generation_results=[],
        evaluation_report=EvaluationReport(
            metrics={"exact_match": MockAggregate(1.0, 5)}, records=[], failures=[]
        ),
        failures=[],
    )

    tracker.log_results(report)
    mock_wandb.summary.update.assert_called_once_with(
        {
            "total_samples": 5,
            "successful_generations": None,
            "failed_generations": None,
            "evaluation_failures": None,
            "exact_match_mean": 1.0,
        }
    )
