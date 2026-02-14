import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from themis.utils.logging_utils import (
    JSONFormatter,
    TraceContextFilter,
    configure_logging,
    get_log_format,
)


class TestJSONFormatter(unittest.TestCase):
    def test_format_structure(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.extra = {"custom_field": "custom_value"}

        formatted = formatter.format(record)
        data = json.loads(formatted)

        self.assertEqual(data["message"], "test message")
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test_logger")
        self.assertEqual(data["custom_field"], "custom_value")
        self.assertTrue("timestamp" in data)

    def test_trace_context(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.trace_id = "trace-123"
        record.span_id = "span-456"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        self.assertEqual(data["trace_id"], "trace-123")
        self.assertEqual(data["span_id"], "span-456")


class TestTraceContextFilter(unittest.TestCase):
    @patch("themis.utils.tracing.get_current_span")
    def test_filter_injects_span(self, mock_get_span):
        # Setup mock span
        mock_span = MagicMock()
        mock_span.span_id = "span-abc"
        mock_get_span.return_value = mock_span

        # Test filter
        log_filter = TraceContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )

        log_filter.filter(record)

        self.assertEqual(record.span_id, "span-abc")


class TestConfigureLogging(unittest.TestCase):
    def test_get_log_format(self):
        configure_logging(log_format="json")
        self.assertEqual(get_log_format(), "json")

        configure_logging(log_format="human")
        self.assertEqual(get_log_format(), "human")
