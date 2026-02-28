"""Tests for extracted api helper functions."""

from themis.core.entities import SamplingConfig


class TestSamplingConfigFromParams:
    """Tests for the new SamplingConfig.from_params() classmethod."""

    def test_from_params_defaults(self):
        config = SamplingConfig.from_params()
        assert config.temperature == 0.0
        assert config.top_p == 0.95
        assert config.max_tokens == 2048

    def test_from_params_custom(self):
        config = SamplingConfig.from_params(temperature=0.7, top_p=0.9, max_tokens=512)
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 512

    def test_from_params_partial(self):
        config = SamplingConfig.from_params(temperature=0.5)
        assert config.temperature == 0.5
        assert config.top_p == 0.95  # default
        assert config.max_tokens == 2048  # default


class TestResolveStorage:
    """Tests for the new public resolve_storage function."""

    def test_resolve_with_path_creates_storage(self, tmp_path):
        from themis.storage import resolve_storage

        storage = resolve_storage(storage_path=tmp_path)
        assert storage is not None

    def test_resolve_with_none_returns_none(self):
        from themis.storage import resolve_storage

        storage = resolve_storage(storage_path=None)
        assert storage is None

    def test_resolve_with_backend_uses_backend(self, tmp_path):
        from unittest.mock import MagicMock
        from themis.storage import resolve_storage

        mock_backend = MagicMock()
        storage = resolve_storage(storage_path=tmp_path, storage_backend=mock_backend)
        assert storage is not None


class TestParseModelPublic:
    """Tests for the new public parse_model in providers/."""

    def test_parse_plain_model(self):
        from themis.providers import parse_model

        provider, model_id, options = parse_model("gpt-4")
        assert provider == "litellm"
        assert model_id == "gpt-4"

    def test_parse_provider_prefix(self):
        from themis.providers import parse_model

        provider, model_id, options = parse_model("fake:fake-math-llm")
        assert provider == "fake"
        assert model_id == "fake-math-llm"

    def test_parse_with_options(self):
        from themis.providers import parse_model

        provider, model_id, options = parse_model("gpt-4", api_key="sk-test")
        assert options["api_key"] == "sk-test"
