"""End-to-end tests for themis.evaluate() API with actual execution."""

import tempfile
from pathlib import Path

import pytest

import themis


class TestEvaluateE2E:
    """End-to-end tests for evaluate() function."""
    
    def test_evaluate_demo_benchmark(self):
        """Test evaluation on demo benchmark with fake model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            # Run evaluation
            report = themis.evaluate(
                "demo",
                model="fake-math-llm",
                limit=2,
                storage=storage_path,
                run_id="test-demo",
                resume=False,
            )
            
            # Verify results
            assert report is not None
            assert len(report.generation_results) == 2
            assert report.evaluation_report is not None
            
            # Verify metrics were computed
            assert len(report.evaluation_report.aggregates) > 0
            
            # Verify storage was created
            assert storage_path.exists()
    
    def test_evaluate_custom_dataset(self):
        """Test evaluation on custom dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            # Create custom dataset
            dataset = [
                {"id": "q1", "question": "What is 2+2?", "answer": "4"},
                {"id": "q2", "question": "What is 5+5?", "answer": "10"},
            ]
            
            # Run evaluation
            report = themis.evaluate(
                dataset=dataset,
                model="fake-math-llm",
                prompt="Q: {question}\nA:",
                storage=storage_path,
                run_id="test-custom",
                resume=False,
            )
            
            # Verify results
            assert report is not None
            assert len(report.generation_results) == 2
            
            # Check that all records have outputs
            for record in report.generation_results:
                assert record.output is not None or record.error is not None
    
    def test_evaluate_with_limit(self):
        """Test that limit parameter works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            # Run with limit
            report = themis.evaluate(
                "demo",
                model="fake-math-llm",
                limit=1,
                storage=storage_path,
                run_id="test-limit",
                resume=False,
            )
            
            # Verify only 1 sample was evaluated
            assert len(report.generation_results) == 1
    
    def test_evaluate_resume_from_cache(self):
        """Test that resume works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            run_id = "test-resume"
            
            # First run
            report1 = themis.evaluate(
                "demo",
                model="fake-math-llm",
                limit=2,
                storage=storage_path,
                run_id=run_id,
                resume=False,
            )
            
            first_results = len(report1.generation_results)
            
            # Second run with resume
            report2 = themis.evaluate(
                "demo",
                model="fake-math-llm",
                limit=2,
                storage=storage_path,
                run_id=run_id,
                resume=True,
            )
            
            # Should have same results (from cache)
            assert len(report2.generation_results) == first_results
    
    def test_evaluate_custom_temperature(self):
        """Test evaluation with custom temperature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            # Run with custom temperature
            report = themis.evaluate(
                "demo",
                model="fake-math-llm",
                limit=1,
                temperature=0.7,
                storage=storage_path,
                run_id="test-temp",
                resume=False,
            )
            
            # Verify task used correct temperature
            assert len(report.generation_results) > 0
            record = report.generation_results[0]
            assert record.task.sampling.temperature == 0.7
    
    def test_evaluate_invalid_benchmark_raises(self):
        """Test that invalid benchmark raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            with pytest.raises(ValueError, match="Unknown benchmark"):
                themis.evaluate(
                    "nonexistent_benchmark",
                    model="fake-math-llm",
                    storage=storage_path,
                    run_id="test-invalid",
                )
    
    def test_evaluate_custom_dataset_requires_prompt(self):
        """Test that custom datasets require a prompt template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test-storage"
            
            dataset = [{"id": "1", "question": "test", "answer": "test"}]
            
            with pytest.raises(ValueError, match="require a prompt template"):
                themis.evaluate(
                    dataset=dataset,
                    model="fake-math-llm",
                    storage=storage_path,
                    run_id="test-no-prompt",
                    # Missing prompt parameter - should fail
                )
