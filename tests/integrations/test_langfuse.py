"""Tests for Langfuse integration tracker."""

from unittest.mock import MagicMock, patch

import pytest

from themis.config.schema import LangfuseConfig


class TestLangfuseTrackerImportHandling:
    """Test Langfuse tracker import and initialization."""

    def test_langfuse_tracker_noop_when_disabled(self):
        """All methods should be no-ops when enable=False."""
        config = LangfuseConfig(enable=False)

        # Import and instantiate with langfuse mocked
        with patch.dict("sys.modules", {"langfuse": MagicMock()}):
            from themis.integrations.langfuse import LangfuseTracker

            tracker = LangfuseTracker(config)

        # All methods should be safe no-ops
        tracker.init({"run_id": "test"})
        tracker.log_metrics({"accuracy": 0.95})
        tracker.log_results(MagicMock())
        tracker.finalize()

    def test_langfuse_config_defaults(self):
        """Config should have sensible defaults."""
        config = LangfuseConfig()
        assert config.enable is False
        assert config.public_key is None
        assert config.secret_key is None
        assert config.base_url is None
        assert config.tags == []
        assert config.trace_name is None
        assert config.enable_tracing is True

    def test_langfuse_config_custom_values(self):
        """Config should accept custom values."""
        config = LangfuseConfig(
            enable=True,
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            base_url="https://my-langfuse.example.com",
            tags=["experiment", "v2"],
            trace_name="my-experiment",
            enable_tracing=False,
        )
        assert config.enable is True
        assert config.public_key == "pk-lf-test"
        assert config.secret_key == "sk-lf-test"
        assert config.base_url == "https://my-langfuse.example.com"
        assert config.tags == ["experiment", "v2"]
        assert config.trace_name == "my-experiment"
        assert config.enable_tracing is False


class TestLangfuseTrackerWithMockedSDK:
    """Test LangfuseTracker behavior with mocked Langfuse SDK."""

    @pytest.fixture
    def mock_langfuse_module(self):
        """Provide a mock langfuse module."""
        mock_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_module.Langfuse.return_value = mock_client_instance

        mock_trace = MagicMock()
        mock_trace.id = "trace-123"
        mock_client_instance.trace.return_value = mock_trace

        return mock_module, mock_client_instance

    def test_init_creates_client_and_trace(self, mock_langfuse_module):
        """init() should create a Langfuse client and root trace."""
        mock_module, mock_client = mock_langfuse_module

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            # Re-import to pick up the mock
            import importlib

            import themis.integrations.langfuse as lf_module

            importlib.reload(lf_module)

            config = LangfuseConfig(
                enable=True,
                public_key="pk-test",
                secret_key="sk-test",
                base_url="https://test.langfuse.com",
                enable_tracing=False,  # Disable litellm callback for simpler test
            )
            tracker = lf_module.LangfuseTracker(config)
            tracker.init({"run_id": "exp-001", "max_samples": 10})

        # Verify client was created with correct kwargs
        mock_module.Langfuse.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            base_url="https://test.langfuse.com",
        )

        # Verify trace was created
        mock_client.trace.assert_called_once_with(
            name="exp-001",
            metadata={"run_id": "exp-001", "max_samples": 10},
            tags=[],
        )

    def test_init_uses_env_vars_when_keys_not_set(self, mock_langfuse_module):
        """init() should not pass keys when not set, letting SDK use env vars."""
        mock_module, mock_client = mock_langfuse_module

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            import importlib

            import themis.integrations.langfuse as lf_module

            importlib.reload(lf_module)

            config = LangfuseConfig(enable=True, enable_tracing=False)
            tracker = lf_module.LangfuseTracker(config)
            tracker.init({"run_id": "test"})

        # Client should be created with no kwargs (SDK reads env vars)
        mock_module.Langfuse.assert_called_once_with()

    def test_log_results_sends_scores(self, mock_langfuse_module):
        """log_results() should send metric scores to Langfuse."""
        mock_module, mock_client = mock_langfuse_module

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            import importlib

            import themis.integrations.langfuse as lf_module

            importlib.reload(lf_module)

            config = LangfuseConfig(enable=True, enable_tracing=False)
            tracker = lf_module.LangfuseTracker(config)
            tracker.init({"run_id": "test"})

        # Create a mock report
        mock_report = MagicMock()
        mock_report.metadata = {
            "total_samples": 100,
            "successful_generations": 95,
            "failed_generations": 5,
            "evaluation_failures": 2,
        }
        mock_aggregate = MagicMock()
        mock_aggregate.mean = 0.85
        mock_aggregate.count = 100
        mock_report.evaluation_report.metrics = {"accuracy": mock_aggregate}

        tracker.log_results(mock_report)

        # Verify score was logged
        mock_client.score.assert_called_once_with(
            trace_id="trace-123",
            name="accuracy_mean",
            value=0.85,
            comment="count=100",
        )

    def test_finalize_flushes(self, mock_langfuse_module):
        """finalize() should flush the Langfuse client."""
        mock_module, mock_client = mock_langfuse_module

        with patch.dict("sys.modules", {"langfuse": mock_module}):
            import importlib

            import themis.integrations.langfuse as lf_module

            importlib.reload(lf_module)

            config = LangfuseConfig(enable=True, enable_tracing=False)
            tracker = lf_module.LangfuseTracker(config)
            tracker.init({"run_id": "test"})

        tracker.finalize()
        mock_client.flush.assert_called_once()
