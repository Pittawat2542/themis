"""Tests for BenchmarkSpec.simple() convenience factory."""

from __future__ import annotations


from themis import BenchmarkSpec
from themis.types.enums import DatasetSource


class TestBenchmarkSpecSimple:
    def test_simple_returns_valid_benchmark_spec(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="quick-eval",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="my-dataset",
            prompt="Solve: {item.question}",
            metric="exact_match",
        )
        assert isinstance(benchmark, BenchmarkSpec)

    def test_simple_sets_benchmark_id(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="my-benchmark",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert benchmark.benchmark_id == "my-benchmark"

    def test_simple_has_one_model(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="claude-3",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert len(benchmark.models) == 1
        assert benchmark.models[0].model_id == "claude-3"

    def test_simple_default_provider_is_openai(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert benchmark.models[0].provider == "openai"

    def test_simple_custom_provider(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="claude-3-sonnet",
            provider="anthropic",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert benchmark.models[0].provider == "anthropic"

    def test_simple_has_one_slice(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert len(benchmark.slices) == 1

    def test_simple_has_one_prompt_variant(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert len(benchmark.prompt_variants) == 1

    def test_simple_prompt_content_in_user_message(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Answer: {item.question}",
            metric="em",
        )
        messages = benchmark.prompt_variants[0].messages
        user_messages = [m for m in messages if m.role.value == "user"]
        assert any("Answer: {item.question}" in m.content for m in user_messages)

    def test_simple_metric_in_score_spec(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="exact_match",
        )
        scores = benchmark.slices[0].scores
        assert len(scores) == 1
        assert "exact_match" in scores[0].metrics

    def test_simple_uses_memory_source_when_specified(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="my-data",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert benchmark.slices[0].dataset.source == DatasetSource.MEMORY

    def test_simple_custom_slice_id(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="b",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
            slice_id="my-slice",
        )
        assert benchmark.slices[0].slice_id == "my-slice"

    def test_simple_default_slice_id_matches_benchmark_id(self) -> None:
        benchmark = BenchmarkSpec.simple(
            benchmark_id="my-eval",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        assert benchmark.slices[0].slice_id == "my-eval"

    def test_simple_result_is_runnable_benchmark(self) -> None:
        """Benchmark created by simple() should pass BenchmarkSpec validation."""
        benchmark = BenchmarkSpec.simple(
            benchmark_id="runnable",
            model_id="gpt-4o",
            dataset_source=DatasetSource.MEMORY,
            dataset_id="ds",
            prompt="Q: {item.q}",
            metric="em",
        )
        # Accessing spec_hash triggers internal validation
        assert benchmark.spec_hash is not None
