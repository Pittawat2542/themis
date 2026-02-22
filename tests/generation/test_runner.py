import unittest
from unittest.mock import MagicMock
from themis.generation.runner import GenerationRunner
from themis.core import entities as core_entities


class TestGenerationRunner(unittest.TestCase):
    def setUp(self):
        self.mock_executor = MagicMock()
        self.mock_executor.reset_mock()
        self.runner = GenerationRunner(executor=self.mock_executor)

        # Setup common test objects
        self.task = core_entities.GenerationTask(
            prompt=core_entities.PromptRender(
                spec=core_entities.PromptSpec(name="test", template=""),
                text="test prompt",
            ),
            model=core_entities.ModelSpec(
                identifier="test-model", provider="test-provider"
            ),
            sampling=core_entities.SamplingConfig(
                temperature=0.0, top_p=1.0, max_tokens=100
            ),
        )

    def test_run_executes_single_task(self):
        """Test basic single task execution."""
        # Setup successful response
        self.mock_executor.execute.return_value = core_entities.GenerationRecord(
            task=self.task,
            output=core_entities.ModelOutput(text="response"),
            error=None,
        )

        results = list(self.runner.run([self.task]))

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].error)
        self.assertEqual(results[0].output.text, "response")
        self.mock_executor.execute.assert_called_once()

    def test_run_handles_provider_error(self):
        """Test handling of provider errors."""
        # Setup error record return (provider catches exceptions and returns record with error)
        error_record = core_entities.GenerationRecord(
            task=self.task,
            output=None,
            error=core_entities.ModelError(message="API Error", kind="api_error"),
        )
        self.mock_executor.execute.return_value = error_record

        # Configure runner to not retry for faster test
        self.runner = GenerationRunner(executor=self.mock_executor, max_retries=0)

        results = list(self.runner.run([self.task]))

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].error)
        self.assertIn("Generation failed", results[0].error.message)
        self.assertIn("API Error", results[0].error.message)

    def test_parallel_execution(self):
        """Test parallel execution mode."""
        # Clean up existing runner as we need one with parallel config
        self.runner = GenerationRunner(executor=self.mock_executor, max_parallel=2)

        tasks = [self.task for _ in range(5)]

        # Configure mock to return valid outputs for parallel execution
        self.mock_executor.execute.return_value = core_entities.GenerationRecord(
            task=self.task,
            output=core_entities.ModelOutput(text="response"),
            error=None,
        )

        results = list(self.runner.run(tasks))

        self.assertEqual(len(results), 5)
        self.assertEqual(self.mock_executor.execute.call_count, 5)
