"""Property-based tests for statistics modules."""

from hypothesis import given, settings, strategies as st
from math import isclose

from themis.evaluation.statistics import comparison_tests as statistics


# Strategies for datasets
# Generate lists of floats that are not too extreme to avoid infinity/NaN issues in basic math
valid_floats = st.floats(
    min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
)
datasets = st.lists(valid_floats, min_size=5, max_size=100)


@settings(deadline=500, max_examples=50)
@given(
    a=st.lists(valid_floats, min_size=5, max_size=50),
    b=st.lists(valid_floats, min_size=5, max_size=50),
)
def test_independent_ttest_symmetry(a, b):
    """Test that independent t-test is symmetric (p-value identical, t-stat inverted)."""
    # Exclude cases where variance is exactly 0
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        return

    res_ab = statistics.t_test(a, b, paired=False)
    res_ba = statistics.t_test(b, a, paired=False)

    # p-value should be identical
    if res_ab.p_value is not None and res_ba.p_value is not None:
        assert isclose(res_ab.p_value, res_ba.p_value, rel_tol=1e-5, abs_tol=1e-9)

    # t-statistic should be perfectly inverted
    assert isclose(res_ab.statistic, -res_ba.statistic, rel_tol=1e-5, abs_tol=1e-9)


@settings(deadline=500, max_examples=50)
@given(
    a=st.lists(valid_floats, min_size=5, max_size=50),
    b=st.lists(valid_floats, min_size=5, max_size=50),
    shift=st.floats(min_value=-100, max_value=100),
)
def test_independent_ttest_shift_invariance(a, b, shift):
    """Test that independent t-test is invariant to identical shifts."""
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        return

    a_shifted = [x + shift for x in a]
    b_shifted = [x + shift for x in b]

    res_orig = statistics.t_test(a, b, paired=False)
    res_shifted = statistics.t_test(a_shifted, b_shifted, paired=False)

    assert isclose(
        res_orig.statistic, res_shifted.statistic, rel_tol=1e-4, abs_tol=1e-5
    )
    if res_orig.p_value is not None and res_shifted.p_value is not None:
        assert isclose(
            res_orig.p_value, res_shifted.p_value, rel_tol=1e-4, abs_tol=1e-5
        )


@settings(deadline=1000, max_examples=20)
@given(
    a=st.lists(valid_floats, min_size=10, max_size=30),
    b=st.lists(valid_floats, min_size=10, max_size=30),
)
def test_bootstrap_symmetry(a, b):
    """Test that bootstrap CI yields inverted (or symmetric) results for (A, B) vs (B, A)."""
    # Bootstrap has randomness and swapping inputs changes random number generation order.
    # Therefore we use a large sample count and check for approximate symmetry.
    res_ab = statistics.bootstrap_confidence_interval(a, b, n_bootstrap=500, seed=42)
    res_ba = statistics.bootstrap_confidence_interval(b, a, n_bootstrap=500, seed=42)

    # Original statistic is deterministic
    assert isclose(res_ab.statistic, -res_ba.statistic, rel_tol=1e-5, abs_tol=1e-9)
    # The CI bounds should be flipped and negated approximately
    lower_ab, upper_ab = res_ab.confidence_interval
    lower_ba, upper_ba = res_ba.confidence_interval

    # We use a loose tolerance because of Monte Carlo error
    diff = max(abs(lower_ab), abs(upper_ab))
    tol = max(1.0, diff * 0.3)  # 30% relative or 1.0 absolute
    assert isclose(lower_ab, -upper_ba, abs_tol=tol)
    assert isclose(upper_ab, -lower_ba, abs_tol=tol)


@settings(deadline=1000, max_examples=20)
@given(
    a=st.lists(valid_floats, min_size=10, max_size=30),
    b=st.lists(valid_floats, min_size=10, max_size=30),
    shift=st.floats(min_value=-100, max_value=100),
)
def test_bootstrap_shift_invariance(a, b, shift):
    """Test that bootstrap CI is shift-invariant."""
    res_orig = statistics.bootstrap_confidence_interval(a, b, n_bootstrap=100, seed=42)

    a_shifted = [x + shift for x in a]
    b_shifted = [x + shift for x in b]
    res_shifted = statistics.bootstrap_confidence_interval(
        a_shifted, b_shifted, n_bootstrap=100, seed=42
    )

    assert isclose(
        res_orig.statistic, res_shifted.statistic, rel_tol=1e-4, abs_tol=1e-5
    )

    ci_orig = res_orig.confidence_interval
    ci_shifted = res_shifted.confidence_interval
    assert isclose(ci_orig[0], ci_shifted[0], rel_tol=1e-4, abs_tol=1e-2)
    assert isclose(ci_orig[1], ci_shifted[1], rel_tol=1e-4, abs_tol=1e-2)


@settings(deadline=1000, max_examples=20)
@given(
    a=st.lists(valid_floats, min_size=10, max_size=30),
    b=st.lists(valid_floats, min_size=10, max_size=30),
)
def test_permutation_symmetry(a, b):
    """Test that permutation test treats groups symmetrically for the absolute statistic."""
    res_ab = statistics.permutation_test(a, b, n_permutations=500, seed=42)
    res_ba = statistics.permutation_test(b, a, n_permutations=500, seed=42)

    assert isclose(res_ab.statistic, res_ba.statistic, rel_tol=1e-5, abs_tol=1e-9)
    # P-value should be identical up to Monte Carlo error
    if res_ab.p_value is not None and res_ba.p_value is not None:
        assert isclose(res_ab.p_value, res_ba.p_value, abs_tol=0.15)
