"""Tests for the unified themis.evaluate() API."""

import pytest

from themis.api import evaluate, _resolve_metrics
from themis.generation.clients import FakeMathModelClient
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
            "gpt-4", base_url="http://localhost:1234/v1", api_key="test-key"
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

    def test_evaluate_custom_dataset(self, tmp_path):
        """Test evaluation with custom dataset runs end-to-end."""
        [
            {"id": "1", "question": "What is 2+2?", "answer": "4"},
        ]

        report = evaluate(
            [
                {"id": "1", "question": "What is 2+2?", "answer": "4"},
            ],
            model="fake-math-llm",
            prompt="What is {question}?",
            storage=tmp_path,
            run_id="custom-dataset-test",
            resume=False,
        )
        assert len(report.generation_results) == 1
        assert "ExactMatch" in report.evaluation_report.metrics

    def test_evaluate_requires_model(self):
        """Test that model parameter is required."""
        with pytest.raises(
            TypeError, match="missing 1 required keyword-only argument: 'model'"
        ):
            evaluate("demo")  # type: ignore[call-arg]

    def test_evaluate_with_invalid_benchmark_raises(self):
        """Test that invalid benchmark raises ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            evaluate("nonexistent-benchmark", model="fake-math-llm")

    def test_evaluate_num_samples_generates_attempts(self, tmp_path):
        """Test that num_samples triggers real repeated sampling attempts."""
        [
            {"id": "1", "question": "2+2", "answer": "4"},
        ]

        report = evaluate(
            [
                {"id": "1", "question": "2+2", "answer": "4"},
            ],
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
        assert [
            attempt.task.metadata.get("attempts") for attempt in record.attempts
        ] == [
            0,
            1,
            2,
        ]

    def test_evaluate_passes_provider_options(self, tmp_path, monkeypatch):
        """Test that provider kwargs are propagated to provider creation."""
        captured: dict[str, object] = {}

        def _fake_create_provider(name: str, **options):
            captured["name"] = name
            captured["options"] = dict(options)
            return FakeMathModelClient(seed=7)

        monkeypatch.setattr("themis.api.create_provider", _fake_create_provider)

        evaluate(
            [{"id": "1", "question": "2+2", "answer": "4"}],
            model="litellm:gpt-4",
            prompt="What is {question}?",
            api_key="test-key",
            api_base="http://localhost:1234/v1",
            storage=tmp_path,
            run_id="provider-options-test",
            resume=False,
        )

        assert captured["name"] == "litellm"
        options = captured["options"]
        assert isinstance(options, dict)
        assert options.get("api_key") == "test-key"
        assert options.get("api_base") == "http://localhost:1234/v1"

    def test_evaluate_normalizes_base_url_alias(self, tmp_path, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_create_provider(name: str, **options):
            captured["name"] = name
            captured["options"] = dict(options)
            return FakeMathModelClient(seed=7)

        monkeypatch.setattr("themis.api.create_provider", _fake_create_provider)

        evaluate(
            [{"id": "1", "question": "2+2", "answer": "4"}],
            model="gpt-4",
            prompt="What is {question}?",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            storage=tmp_path,
            run_id="provider-base-url-alias",
            resume=False,
        )

        assert captured["name"] == "litellm"
        options = captured["options"]
        assert isinstance(options, dict)
        assert options.get("api_base") == "http://localhost:1234/v1"
        assert "base_url" not in options

    def test_evaluate_calls_on_result_callback(self, tmp_path):
        """Test that on_result callback is called for each generation record."""
        seen_ids: list[str | None] = []

        def _on_result(record):
            seen_ids.append(record.task.metadata.get("dataset_id"))

        evaluate(
            [
                {"id": "1", "question": "2+2", "answer": "4"},
                {"id": "2", "question": "1+1", "answer": "2"},
            ],
            model="fake-math-llm",
            prompt="What is {question}?",
            on_result=_on_result,
            storage=tmp_path,
            run_id="on-result-callback-test",
            resume=False,
        )

        assert len(seen_ids) == 2
        assert set(seen_ids) == {"1", "2"}

    def test_evaluate_rejects_unknown_distributed_arg(self, tmp_path):
        """Test that removed placeholder args fail at API boundary."""
        with pytest.raises(ValueError, match="Unsupported option"):
            evaluate(
                [{"id": "1", "question": "2+2", "answer": "4"}],
                model="fake-math-llm",
                prompt="What is {question}?",
                distributed=True,
                storage=tmp_path,
                run_id="distributed-not-supported",
                resume=False,
            )

    def test_evaluate_custom_dataset_reference_field(self, tmp_path):
        """Test that custom datasets support `reference` field as ground truth."""
        report = evaluate(
            [{"id": "1", "question": "2+2", "reference": "4"}],
            model="fake-math-llm",
            prompt="What is {question}?",
            metrics=["exact_match"],
            storage=tmp_path,
            run_id="reference-field-test",
            resume=False,
        )
        assert report.evaluation_report.failures == []

    def test_evaluate_custom_dataset_reference_field_override(self, tmp_path):
        report = evaluate(
            [{"id": "1", "question": "2+2", "gold_label": "4"}],
            model="fake-math-llm",
            prompt="What is {question}?",
            reference_field="gold_label",
            metrics=["exact_match"],
            storage=tmp_path,
            run_id="reference-field-override-test",
            resume=False,
        )
        assert report.evaluation_report.failures == []

    def test_evaluate_custom_dataset_mixed_reference_fields_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="mixed or partial reference fields"):
            evaluate(
                [
                    {"id": "1", "question": "2+2", "answer": "4"},
                    {"id": "2", "question": "1+1", "reference": "2"},
                ],
                model="fake-math-llm",
                prompt="What is {question}?",
                metrics=["exact_match"],
                storage=tmp_path,
                run_id="mixed-reference-fields",
                resume=False,
            )

    def test_evaluate_custom_dataset_missing_reference_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="Could not detect a reference field"):
            evaluate(
                [{"id": "1", "question": "2+2"}],
                model="fake-math-llm",
                prompt="What is {question}?",
                metrics=["exact_match"],
                storage=tmp_path,
                run_id="missing-reference-fast-fail",
                resume=False,
            )

    def test_evaluate_supports_bounded_memory_mode(self, tmp_path):
        report = evaluate(
            [
                {"id": "1", "question": "2+2", "answer": "4"},
                {"id": "2", "question": "1+1", "answer": "2"},
                {"id": "3", "question": "3+3", "answer": "6"},
            ],
            model="fake-math-llm",
            prompt="What is {question}?",
            metrics=["exact_match"],
            storage=tmp_path,
            run_id="bounded-memory-test",
            resume=False,
            max_records_in_memory=1,
        )

        assert report.metadata["generation_records_retained"] == 1
        assert report.metadata["generation_records_dropped"] == 2
        assert report.metadata["evaluation_records_retained"] == 1
        assert report.metadata["evaluation_records_dropped"] == 2
        assert report.evaluation_report.metrics["ExactMatch"].count == 3

    def test_evaluate_reproducible_with_fixed_seed_and_manifest(self, tmp_path):
        [
            {"id": "1", "question": "2+2", "answer": "4"},
            {"id": "2", "question": "1+1", "answer": "2"},
        ]

        report_a = evaluate(
            [
                {"id": "1", "question": "2+2", "answer": "4"},
                {"id": "2", "question": "1+1", "answer": "2"},
            ],
            model="fake:fake-math-llm",
            prompt="What is {question}?",
            metrics=["exact_match"],
            storage=tmp_path,
            run_id="repro-a",
            resume=False,
            seed=123,
        )
        report_b = evaluate(
            [
                {"id": "1", "question": "2+2", "answer": "4"},
                {"id": "2", "question": "1+1", "answer": "2"},
            ],
            model="fake:fake-math-llm",
            prompt="What is {question}?",
            metrics=["exact_match"],
            storage=tmp_path,
            run_id="repro-b",
            resume=False,
            seed=123,
        )

        assert report_a.metadata["manifest_hash"] == report_b.metadata["manifest_hash"]
        assert report_a.metadata["reproducibility_manifest"]["seeds"] == {
            "provider_seed": 123,
            "sampling_seed": None,
        }
        outputs_a = {
            r.task.metadata.get("dataset_id"): r.output.text
            for r in report_a.generation_results
            if r.output is not None
        }
        outputs_b = {
            r.task.metadata.get("dataset_id"): r.output.text
            for r in report_b.generation_results
            if r.output is not None
        }
        assert outputs_a == outputs_b
        assert report_a.evaluation_report.metrics["ExactMatch"].mean == pytest.approx(
            report_b.evaluation_report.metrics["ExactMatch"].mean
        )


# Run simple import test to verify module loads
def test_import():
    """Test that module can be imported."""
    import themis

    assert hasattr(themis, "evaluate")
