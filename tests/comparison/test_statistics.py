"""Tests for comparison statistics module."""

import random

import pytest

from themis.comparison import statistics as comparison_statistics
from themis.comparison.statistics import (
    StatisticalTest,
    StatisticalTestResult,
    bootstrap_confidence_interval,
    mcnemar_test,
    permutation_test,
    t_test,
)


class TestTTest:
    """Tests for t-test."""
    
    def test_t_test_paired_significant_difference(self):
        """Test paired t-test with significant difference."""
        samples_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        samples_b = [0.5, 1.5, 2.5, 3.5, 4.5]  # Consistently 0.5 lower
        
        result = t_test(samples_a, samples_b, paired=True, alpha=0.05)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "t-test (paired)"
        assert result.statistic > 0
        assert result.effect_size is not None
    
    def test_t_test_paired_no_difference(self):
        """Test paired t-test with no difference."""
        samples_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        samples_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = t_test(samples_a, samples_b, paired=True)
        
        assert result.statistic == 0.0
        assert not result.significant
    
    def test_t_test_independent(self):
        """Test independent samples t-test."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [4.0, 5.0, 6.0]
        
        result = t_test(samples_a, samples_b, paired=False)
        
        assert result.test_name == "t-test (independent)"
        assert result.confidence_interval is not None
    
    def test_t_test_empty_samples_raises(self):
        """Test that empty samples raise ValueError."""
        with pytest.raises(ValueError, match="empty samples"):
            t_test([], [1.0, 2.0])
    
    def test_t_test_mismatched_lengths_paired_raises(self):
        """Test that mismatched lengths raise for paired test."""
        with pytest.raises(ValueError, match="equal sample sizes"):
            t_test([1.0, 2.0], [1.0, 2.0, 3.0], paired=True)

    def test_t_test_p_value_monotonic_for_small_df(self):
        """Test p-values are monotonic with larger t-statistics for same df."""
        p_low = comparison_statistics._approximate_t_test_p_value(2.1, 10)
        p_high = comparison_statistics._approximate_t_test_p_value(2.9, 10)
        assert p_high < p_low


class TestBootstrap:
    """Tests for bootstrap confidence interval."""
    
    def test_bootstrap_significant_difference(self):
        """Test bootstrap with significant difference."""
        samples_a = [5.0, 6.0, 7.0, 8.0, 9.0]
        samples_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = bootstrap_confidence_interval(
            samples_a, samples_b, n_bootstrap=1000, seed=42
        )
        
        assert isinstance(result, StatisticalTestResult)
        assert "bootstrap" in result.test_name
        assert result.confidence_interval is not None
        assert result.significant  # Large difference should be significant
    
    def test_bootstrap_no_difference(self):
        """Test bootstrap with no difference."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [1.0, 2.0, 3.0]
        
        result = bootstrap_confidence_interval(
            samples_a, samples_b, n_bootstrap=100, seed=42
        )
        
        assert not result.significant
        ci = result.confidence_interval
        assert ci[0] <= 0 <= ci[1]  # 0 should be in CI
    
    def test_bootstrap_custom_statistic(self):
        """Test bootstrap with custom statistic function."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [2.0, 3.0, 4.0]
        
        # Custom statistic: median difference
        def median_diff(a, b):
            return sorted(a)[len(a)//2] - sorted(b)[len(b)//2]
        
        result = bootstrap_confidence_interval(
            samples_a, samples_b,
            n_bootstrap=100,
            statistic_fn=median_diff,
            seed=42,
        )
        
        assert result.statistic == median_diff(samples_a, samples_b)

    def test_bootstrap_does_not_mutate_global_random_state(self):
        """Test bootstrap uses local RNG and does not reset global random state."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [2.0, 3.0, 4.0]

        random.seed(2024)
        _ = random.random()
        expected_next = random.random()

        random.seed(2024)
        _ = random.random()
        bootstrap_confidence_interval(samples_a, samples_b, n_bootstrap=100, seed=42)
        observed_next = random.random()

        assert observed_next == expected_next


class TestPermutation:
    """Tests for permutation test."""
    
    def test_permutation_significant_difference(self):
        """Test permutation test with significant difference."""
        samples_a = [10.0, 11.0, 12.0, 13.0, 14.0]
        samples_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = permutation_test(
            samples_a, samples_b, n_permutations=1000, seed=42
        )
        
        assert isinstance(result, StatisticalTestResult)
        assert "permutation" in result.test_name
        assert result.significant
    
    def test_permutation_no_difference(self):
        """Test permutation test with no difference."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [1.5, 2.5, 3.5]
        
        result = permutation_test(
            samples_a, samples_b, n_permutations=100, seed=42
        )
        
        # Small difference shouldn't be significant
        assert result.p_value > 0.01
    
    def test_permutation_custom_statistic(self):
        """Test permutation with custom statistic."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [2.0, 3.0, 4.0]
        
        def max_diff(a, b):
            return abs(max(a) - max(b))  # Use absolute value
        
        result = permutation_test(
            samples_a, samples_b,
            statistic_fn=max_diff,
            n_permutations=100,
            seed=42,
        )
        
        assert result.statistic >= 0

    def test_permutation_does_not_mutate_global_random_state(self):
        """Test permutation uses local RNG and does not reset global random state."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [2.0, 3.0, 4.0]

        random.seed(12345)
        _ = random.random()
        expected_next = random.random()

        random.seed(12345)
        _ = random.random()
        permutation_test(samples_a, samples_b, n_permutations=100, seed=42)
        observed_next = random.random()

        assert observed_next == expected_next


class TestMcNemar:
    """Tests for McNemar's test."""
    
    def test_mcnemar_significant(self):
        """Test McNemar's test with significant difference."""
        # A correct, B wrong: 20
        # A wrong, B correct: 5
        contingency = (10, 20, 5, 15)
        
        result = mcnemar_test(contingency)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "McNemar's test"
        assert result.significant
    
    def test_mcnemar_no_discordant(self):
        """Test McNemar's test with no discordant pairs."""
        # All concordant
        contingency = (10, 0, 0, 15)
        
        result = mcnemar_test(contingency)
        
        assert not result.significant
        assert result.p_value == 1.0
    
    def test_mcnemar_balanced_discordant(self):
        """Test McNemar's test with balanced discordant pairs."""
        # Equal A>B and B>A
        contingency = (10, 10, 10, 10)
        
        result = mcnemar_test(contingency)
        
        assert not result.significant

    def test_mcnemar_uses_exact_binomial_p_value(self):
        """Test McNemar p-value uses exact binomial test for discordant pairs."""
        # b=10, c=0 -> two-sided exact p-value = 2 * (0.5 ** 10) = 0.001953125
        contingency = (0, 10, 0, 0)
        result = mcnemar_test(contingency)
        assert result.p_value == pytest.approx(0.001953125, rel=1e-6)


class TestTestResult:
    """Tests for TestResult class."""
    
    def test_test_result_str(self):
        """Test StatisticalTestResult string representation."""
        result = StatisticalTestResult(
            test_name="test",
            statistic=2.5,
            p_value=0.05,
            significant=True,
            effect_size=0.8,
            confidence_interval=(0.1, 0.9),
        )
        
        str_repr = str(result)
        assert "test" in str_repr
        assert "p=0.0500" in str_repr
        assert "significant" in str_repr
        assert "effect_size=0.800" in str_repr
        assert "CI=[0.100, 0.900]" in str_repr
    
    def test_test_result_minimal(self):
        """Test StatisticalTestResult with minimal fields."""
        result = StatisticalTestResult(
            test_name="minimal",
            statistic=1.0,
            p_value=0.1,
            significant=False,
        )
        
        str_repr = str(result)
        assert "minimal" in str_repr
        assert "not significant" in str_repr
