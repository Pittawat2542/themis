"""Integration tests for end-to-end evaluation workflows."""

from pathlib import Path

import pytest

from themis import evaluate
from themis.comparison import compare_runs
from themis.core.entities import ExperimentReport


class TestEvaluationWorkflow:
    """Test complete evaluation workflows."""

    def test_simple_evaluation(self, tmp_path):
        """Test basic evaluation with demo benchmark."""
        storage = tmp_path / "storage"
        
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=5,
            storage=storage,
            run_id="test-simple",
        )
        
        assert isinstance(result, ExperimentReport)
        assert result.num_samples == 5
        assert "exact_match" in result.metrics
        assert result.metrics["exact_match"] > 0.0

    def test_evaluation_with_resume(self, tmp_path):
        """Test evaluation with caching and resume."""
        storage = tmp_path / "storage"
        run_id = "test-resume"
        
        # First run - generate some results
        result1 = evaluate(
            "demo",
            model="fake-math-llm",
            limit=3,
            storage=storage,
            run_id=run_id,
            resume=True,
        )
        
        assert result1.num_samples == 3
        
        # Second run - should use cache
        result2 = evaluate(
            "demo",
            model="fake-math-llm",
            limit=3,
            storage=storage,
            run_id=run_id,
            resume=True,
        )
        
        assert result2.num_samples == 3
        # Results should be identical (from cache)
        assert result2.metrics["exact_match"] == result1.metrics["exact_match"]

    def test_evaluation_with_custom_dataset(self, tmp_path):
        """Test evaluation with custom dataset."""
        storage = tmp_path / "storage"
        
        dataset = [
            {"id": "1", "prompt": "What is 2+2?", "answer": "4"},
            {"id": "2", "prompt": "What is 5-3?", "answer": "2"},
            {"id": "3", "prompt": "What is 3*4?", "answer": "12"},
        ]
        
        result = evaluate(
            dataset,
            model="fake-math-llm",
            prompt="{prompt}",
            metrics=["exact_match"],
            storage=storage,
            run_id="test-custom",
        )
        
        assert result.num_samples == 3
        assert "exact_match" in result.metrics

    def test_evaluation_different_models(self, tmp_path):
        """Test evaluation with multiple models."""
        storage = tmp_path / "storage"
        
        models = ["fake-math-llm", "fake-math-llm"]
        run_ids = ["model-1", "model-2"]
        
        results = []
        for model, run_id in zip(models, run_ids):
            result = evaluate(
                benchmark="demo",
                model=model,
                limit=3,
                storage=storage,
                run_id=run_id,
            )
            results.append(result)
        
        assert len(results) == 2
        assert all(r.num_samples == 3 for r in results)

    def test_evaluation_with_temperature(self, tmp_path):
        """Test evaluation with different sampling parameters."""
        storage = tmp_path / "storage"
        
        temperatures = [0.0, 0.5]
        
        for i, temp in enumerate(temperatures):
            result = evaluate(
                benchmark="demo",
                model="fake-math-llm",
                limit=3,
                temperature=temp,
                storage=storage,
                run_id=f"temp-{temp}",
            )
            
            assert result.num_samples == 3
            assert "exact_match" in result.metrics

    def test_evaluation_multiple_samples(self, tmp_path):
        """Test evaluation generating multiple samples per prompt."""
        storage = tmp_path / "storage"
        
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=2,
            num_samples=3,  # Generate 3 responses per prompt
            storage=storage,
            run_id="test-multi-sample",
        )
        
        assert result.num_samples == 2  # Number of prompts
        # Should have generated multiple responses per prompt

    def test_evaluation_with_workers(self, tmp_path):
        """Test evaluation with parallel workers."""
        storage = tmp_path / "storage"
        
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=5,
            workers=4,
            storage=storage,
            run_id="test-workers",
        )
        
        assert result.num_samples == 5


class TestComparisonWorkflow:
    """Test comparison workflows."""

    def test_compare_two_runs(self, tmp_path):
        """Test comparing two evaluation runs."""
        storage = tmp_path / "storage"
        
        # Run two evaluations
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=5,
            storage=storage,
            run_id="run-a",
        )
        
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=5,
            storage=storage,
            run_id="run-b",
        )
        
        # Compare them
        report = compare_runs(
            ["run-a", "run-b"],
            storage_path=storage,
            statistical_test="bootstrap",
        )
        
        assert report is not None
        assert len(report.pairwise_results) > 0

    def test_compare_multiple_runs(self, tmp_path):
        """Test comparing multiple runs."""
        storage = tmp_path / "storage"
        
        run_ids = ["run-1", "run-2", "run-3"]
        
        # Create multiple runs
        for run_id in run_ids:
            evaluate(
                benchmark="demo",
                model="fake-math-llm",
                limit=3,
                storage=storage,
                run_id=run_id,
            )
        
        # Compare all
        report = compare_runs(
            run_ids,
            storage_path=storage,
        )
        
        assert report is not None
        # Should have pairwise comparisons
        assert len(report.pairwise_results) >= len(run_ids) - 1


class TestStorageWorkflow:
    """Test storage and caching workflows."""

    def test_storage_persistence(self, tmp_path):
        """Test that results are persisted correctly."""
        storage = tmp_path / "storage"
        run_id = "test-persist"
        
        # First evaluation
        result1 = evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=3,
            storage=storage,
            run_id=run_id,
        )
        
        # Verify storage files exist
        run_dir = storage / run_id
        assert run_dir.exists()
        assert (run_dir / "metadata.json").exists()

    def test_storage_resume_after_failure(self, tmp_path):
        """Test resuming after partial completion."""
        storage = tmp_path / "storage"
        run_id = "test-resume-failure"
        
        # First run - partial
        evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=2,
            storage=storage,
            run_id=run_id,
        )
        
        # Second run - complete with resume
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=5,
            storage=storage,
            run_id=run_id,
            resume=True,
        )
        
        # Should complete with 5 samples
        assert result.num_samples == 5

    def test_storage_no_resume(self, tmp_path):
        """Test running without resume (fresh start)."""
        storage = tmp_path / "storage"
        run_id = "test-no-resume"
        
        # First run
        result1 = evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=3,
            storage=storage,
            run_id=run_id,
        )
        
        # Second run without resume - should recompute
        result2 = evaluate(
            benchmark="demo",
            model="fake-math-llm",
            limit=3,
            storage=storage,
            run_id=run_id,
            resume=False,
        )
        
        # Both should complete successfully
        assert result1.num_samples == 3
        assert result2.num_samples == 3


class TestBenchmarkWorkflow:
    """Test workflows with different benchmarks."""

    def test_demo_benchmark(self, tmp_path):
        """Test demo benchmark workflow."""
        storage = tmp_path / "storage"
        
        result = evaluate(
            "demo",
            model="fake-math-llm",
            storage=storage,
        )
        
        assert result.num_samples > 0
        assert "exact_match" in result.metrics

    @pytest.mark.skip("Requires real API key")
    def test_gsm8k_benchmark(self, tmp_path):
        """Test GSM8K benchmark workflow."""
        storage = tmp_path / "storage"
        
        result =         evaluate(
            "gsm8k",
            model="gpt-4",
            limit=10,
            storage=storage,
        )
        
        assert result.num_samples == 10
        assert "exact_match" in result.metrics
        assert "math_verify" in result.metrics


class TestMetricsWorkflow:
    """Test workflows with different metrics."""

    def test_math_metrics(self, tmp_path):
        """Test math-specific metrics."""
        storage = tmp_path / "storage"
        
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=3,
            metrics=["exact_match", "math_verify"],
            storage=storage,
        )
        
        assert "exact_match" in result.metrics
        assert "math_verify" in result.metrics

    @pytest.mark.skip("Requires NLP dependencies")
    def test_nlp_metrics(self, tmp_path):
        """Test NLP metrics."""
        storage = tmp_path / "storage"
        
        dataset = [
            {"id": "1", "prompt": "Hello", "reference": "World"},
            {"id": "2", "prompt": "Test", "reference": "Pass"},
        ]
        
        result = evaluate(
            dataset,
            model="fake-math-llm",
            prompt="{prompt}",
            metrics=["BLEU", "ROUGE"],
            storage=storage,
        )
        
        assert "BLEU" in result.metrics
        assert "ROUGE" in result.metrics


class TestErrorHandling:
    """Test error handling in workflows."""

    def test_invalid_benchmark(self):
        """Test handling of invalid benchmark name."""
        with pytest.raises(ValueError):
            evaluate(
                "invalid-benchmark-name",
                model="fake-math-llm",
            )

    def test_invalid_storage_path(self):
        """Test handling of invalid storage path."""
        # Should handle gracefully or create directory
        result = evaluate(
            "demo",
            model="fake-math-llm",
            limit=1,
            storage="/tmp/themis-test-storage",
            run_id="test-storage",
        )
        
        assert result is not None

    def test_missing_required_params(self):
        """Test handling of missing required parameters."""
        with pytest.raises(TypeError):
            # Missing model parameter
            evaluate(benchmark="demo")
