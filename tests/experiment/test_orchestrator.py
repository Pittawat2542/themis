import unittest
from unittest.mock import MagicMock
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.core import entities as core_entities
from themis.evaluation.reports import EvaluationReport


class TestExperimentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_plan = MagicMock()
        self.mock_runner = MagicMock()
        self.mock_pipeline = MagicMock()
        self.mock_integrations = MagicMock()

        self.mock_integrations = MagicMock()

        self.orchestrator = ExperimentOrchestrator(
            generation_plan=self.mock_plan,
            generation_runner=self.mock_runner,
            evaluation_pipeline=self.mock_pipeline,
            integration_manager=self.mock_integrations,
        )

    def test_run_orchestrates_experiment(self):
        """Test full experiment orchestration flow."""
        # Setup mocks
        mock_task = MagicMock()
        mock_task.prompt.text = "test prompt"
        # Configure metadata to be JSON serializable
        mock_task.metadata = {"dataset_id": "1"}
        # Configure reference to be None to avoid serialization issues
        mock_task.reference = None
        # Configure sampling to support formatting
        mock_task.sampling.temperature = 0.7
        mock_task.sampling.top_p = 1.0
        mock_task.sampling.max_tokens = 100
        # Configure prompt and model for cache key generation
        mock_task.prompt.spec.name = "default_template"
        mock_task.prompt.spec.template = "template"
        mock_task.prompt.context = {}
        mock_task.model.identifier = "gpt-4"
        mock_task.model.provider = "openai"

        self.mock_plan.expand.return_value = [mock_task]
        self.mock_runner.run.return_value = [
            core_entities.GenerationRecord(
                task=mock_task,
                output=core_entities.ModelOutput(text="response"),
                error=None,
            )
        ]
        self.mock_pipeline.evaluate.return_value = EvaluationReport(
            metrics={}, failures=[], records=[], metadata={}
        )

        # Mock add_task to return an ID
        self.orchestrator.run(dataset=[{"id": "1"}])

        # Verify orchestration
        self.mock_integrations.initialize_run.assert_called_once()
        self.mock_plan.expand.assert_called_once()
        self.mock_runner.run.assert_called_once()
        self.mock_pipeline.evaluate.assert_called_once()
        self.mock_integrations.log_results.assert_called_once()

    def test_run_handles_empty_dataset(self):
        """Test orchestration with empty dataset."""
        self.mock_plan.expand.return_value = []

        # Mock pipeline evaluate for empty run
        self.mock_pipeline.evaluate.return_value = EvaluationReport(
            metrics={}, failures=[], records=[], metadata={}
        )

        report = self.orchestrator.run(dataset=[])

        # Runner is still called with empty iterator
        self.mock_runner.run.assert_called_once()
        self.assertEqual(len(report.generation_results), 0)
