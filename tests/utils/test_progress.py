import unittest
import threading
from unittest.mock import MagicMock, patch
from themis.utils.progress import (
    LogProgressReporter,
    WandBProgressReporter,
    CompositeProgressReporter,
)


class TestLogProgressReporter(unittest.TestCase):
    @patch("themis.utils.progress.logger")
    def test_add_task_logs_start(self, mock_logger):
        reporter = LogProgressReporter()
        task_id = reporter.add_task("Test Task", total=100)

        self.assertEqual(task_id, 0)
        mock_logger.info.assert_called()
        args, kwargs = mock_logger.info.call_args
        # args[0] is format string, args[1] is description
        self.assertEqual(args[1], "Test Task")
        self.assertEqual(kwargs["extra"]["event"], "progress_start")
        self.assertEqual(kwargs["extra"]["total"], 100)

    @patch("themis.utils.progress.logger")
    def test_update_debouncing(self, mock_logger):
        reporter = LogProgressReporter()
        task_id = reporter.add_task("Test Task", total=100)

        # Reset mock to clear start log
        mock_logger.reset_mock()

        # Update 1%: Should strictly NOT log (debouncing < 10%)
        reporter.update(task_id, advance=1)
        mock_logger.info.assert_not_called()

        # Update to 10%: Should log
        reporter.update(task_id, advance=9)  # total 10
        mock_logger.info.assert_called()
        args, kwargs = mock_logger.info.call_args
        self.assertEqual(kwargs["extra"]["event"], "progress_update")
        self.assertEqual(kwargs["extra"]["percent"], 10.0)

    @patch("themis.utils.progress.logger")
    def test_unknown_total(self, mock_logger):
        reporter = LogProgressReporter()
        task_id = reporter.add_task("Unknown Total", total=None)
        mock_logger.reset_mock()

        # Should log every 10 for small counts
        reporter.update(task_id, advance=10)
        mock_logger.info.assert_called()

    @patch("themis.utils.progress.logger")
    def test_completion_always_logs(self, mock_logger):
        reporter = LogProgressReporter()
        task_id = reporter.add_task("Test", total=100)
        mock_logger.reset_mock()

        # Jump to 100%
        reporter.update(task_id, advance=100)
        mock_logger.info.assert_called()
        _, kwargs = mock_logger.info.call_args
        self.assertEqual(kwargs["extra"]["percent"], 100.0)

    def test_thread_safety(self):
        # This test ensures no exceptions are raised during concurrent updates
        reporter = LogProgressReporter()
        task_id = reporter.add_task("Concurrent Task", total=1000)

        def worker():
            for _ in range(100):
                reporter.update(task_id, advance=1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify internal state is consistent
        # We need to access private member for verification, which is acceptable in tests
        self.assertEqual(reporter._tasks[task_id]["completed"], 1000)


class TestWandBProgressReporter(unittest.TestCase):
    def test_update_logs_metrics(self):
        mock_integration = MagicMock()
        reporter = WandBProgressReporter(mock_integration)

        reporter.update(0, completed=10, total=100, custom_metric=0.5)

        mock_integration.log_metrics.assert_called()
        args = mock_integration.log_metrics.call_args[0][0]
        self.assertEqual(args["progress/completed"], 10)
        self.assertEqual(args["progress/total"], 100)
        self.assertEqual(args["progress/custom_metric"], 0.5)


class TestCompositeProgressReporter(unittest.TestCase):
    def test_delegates_calls(self):
        mock_reporter1 = MagicMock()
        mock_reporter2 = MagicMock()
        mock_reporter1.add_task.return_value = 1
        mock_reporter2.add_task.return_value = 2

        composite = CompositeProgressReporter([mock_reporter1, mock_reporter2])

        # Test add_task
        tid = composite.add_task("Test")
        mock_reporter1.add_task.assert_called_with("Test", None)
        mock_reporter2.add_task.assert_called_with("Test", None)
        self.assertEqual(tid, 1)  # Should return first ID

        # Test update
        composite.update(tid, advance=5, foo="bar")
        mock_reporter1.update.assert_called_with(tid, 5, foo="bar")
        mock_reporter2.update.assert_called_with(tid, 5, foo="bar")

        # Test context manager
        composite.__enter__()
        mock_reporter1.__enter__.assert_called()
        mock_reporter2.__enter__.assert_called()

        composite.__exit__(None, None, None)
        mock_reporter1.__exit__.assert_called()
        mock_reporter2.__exit__.assert_called()
