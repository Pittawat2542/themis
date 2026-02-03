"""Tests for the unified themis.evaluate() API."""

import pytest

from themis.api import evaluate, _resolve_metrics
from themis.presets import get_benchmark_preset, list_benchmarks, parse_model_name


class TestResolveMetrics:
    """Test metric resolution."""
    
    def test_resolve_exact_match(self):
        """Test resolving exact_match metric."""
        metrics = _resolve_metrics(["exact_match"])
        assert len(metrics) == 1
        assert metrics[0].name == "ExactMatch"
    
    def test_resolve_multiple_metrics(self):
        """Test resolving multiple metrics."""
        metrics = _resolve_metrics(["exact_match", "response_length"])
        assert len(metrics) == 2
        assert any(m.name == "ExactMatch" for m in metrics)
        assert any(m.name == "ResponseLength" for m in metrics)
    
    def test_resolve_unknown_metric_raises(self):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            _resolve_metrics(["nonexistent_metric"])

    def test_resolve_metric_aliases(self):
        """Test resolving CamelCase metric names."""
        metrics = _resolve_metrics(["ExactMatch", "ResponseLength"])
        assert len(metrics) == 2
        assert any(m.name == "ExactMatch" for m in metrics)
        assert any(m.name == "ResponseLength" for m in metrics)


class TestModelParsing:
    """Test model name parsing."""
    
    def test_parse_gpt4(self):
        """Test parsing GPT-4 model name."""
        provider, model_id, options = parse_model_name("gpt-4")
        assert provider == "litellm"
        assert model_id == "gpt-4"
        assert options == {}
    
    def test_parse_claude(self):
        """Test parsing Claude model name."""
        provider, model_id, options = parse_model_name("claude-3-opus-20240229")
        assert provider == "litellm"
        assert model_id == "claude-3-opus-20240229"
    
    def test_parse_with_options(self):
        """Test parsing with additional options."""
        provider, model_id, options = parse_model_name(
            "gpt-4",
            base_url="http://localhost:1234/v1",
            api_key="test-key"
        )
        assert provider == "litellm"
        assert options["base_url"] == "http://localhost:1234/v1"
        assert options["api_key"] == "test-key"
    
    def test_parse_fake_model(self):
        """Test parsing fake model for testing."""
        provider, model_id, options = parse_model_name("fake-math-llm")
        assert provider == "fake"
        assert model_id == "fake-math-llm"


class TestBenchmarkPresets:
    """Test benchmark preset system."""
    
    def test_list_benchmarks(self):
        """Test listing all benchmarks."""
        benchmarks = list_benchmarks()
        assert len(benchmarks) > 0
        assert "demo" in benchmarks
        assert "math500" in benchmarks
        assert "gsm8k" in benchmarks
    
    def test_get_demo_preset(self):
        """Test getting demo benchmark preset."""
        preset = get_benchmark_preset("demo")
        assert preset.name == "demo"
        assert preset.prompt_template is not None
        assert len(preset.metrics) > 0
        assert preset.extractor is not None
    
    def test_get_math500_preset(self):
        """Test getting MATH-500 preset."""
        preset = get_benchmark_preset("math500")
        assert preset.name == "math500"
        assert preset.reference_field == "solution"
        assert preset.dataset_id_field == "unique_id"
    
    def test_get_unknown_preset_raises(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark_preset("nonexistent_benchmark")
    
    def test_demo_dataset_loader(self):
        """Test demo dataset loader."""
        preset = get_benchmark_preset("demo")
        dataset = preset.load_dataset(limit=2)
        assert len(dataset) == 2
        assert all("question" in sample for sample in dataset)
        assert all("answer" in sample for sample in dataset)


class TestEvaluateAPI:
    """Test the main evaluate() API."""
    
    def test_evaluate_custom_dataset(self):
        """Test evaluation with custom dataset."""
        # Note: This would require full integration, so we just test the interface
        dataset = [
            {"id": "1", "question": "What is 2+2?", "answer": "4"},
        ]
        
        # This would require a real model, so we just verify the function signature
        # In a real test, we'd use a mock or fake model
        pass  # TODO: Implement with mocked components
    
    def test_evaluate_requires_model(self):
        """Test that model parameter is required."""
        # The function signature requires model, so this is enforced by Python
        pass
    
    def test_evaluate_with_invalid_benchmark_raises(self):
        """Test that invalid benchmark raises ValueError."""
        # Would need to actually call evaluate() with mocked components
        pass  # TODO: Implement with mocked components

    def test_evaluate_num_samples_generates_attempts(self, tmp_path):
        """Test that num_samples triggers repeated sampling."""
        dataset = [
            {"id": "1", "question": "2+2", "answer": "4"},
        ]

        report = evaluate(
            dataset,
            model="fake-math-llm",
            prompt="What is {question}?",
            num_samples=3,
            storage=tmp_path,
            run_id="num-samples-test",
            resume=False,
        )

        record = report.generation_results[0]
        assert len(record.attempts) == 3
        assert record.metrics.get("attempt_count") == 3


# Run simple import test to verify module loads
def test_import():
    """Test that module can be imported."""
    import themis
    assert hasattr(themis, 'evaluate')
