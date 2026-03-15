import pytest
import numpy as np
import pandas as pd
from themis.stats.stats_engine import StatsEngine, ComparisonResult
from themis.types.enums import PValueCorrection


def test_stats_engine_aggregation():
    # Mock data
    data = pd.DataFrame(
        {
            "candidate_hash": ["h1", "h2", "h3", "h4"],
            "model_id": ["gpt-4", "gpt-4", "gpt-3.5", "gpt-3.5"],
            "task_id": ["math", "math", "math", "math"],
            "item_id": ["i1", "i2", "i1", "i2"],
            "metric_value": [1.0, 1.0, 0.0, 1.0],
        }
    )

    engine = StatsEngine()

    # 1. Test basic aggregation
    agg = engine.aggregate(data, group_by=["model_id"])
    assert agg.loc["gpt-4", "mean"] == 1.0
    assert agg.loc["gpt-3.5", "mean"] == 0.5


def test_stats_engine_paired_bootstrap():
    data = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4", "i5", "i6", "i7"],
            "gpt-4": [1.0, 1.0, 0.8, 1.0, 0.9, 1.0, 0.9],
            "gpt-3.5": [0.8, 0.5, 0.5, 0.9, 0.7, 0.4, 0.5],
        }
    )

    engine = StatsEngine()

    # Test paired delta (gpt-4 vs gpt-3.5)
    result = engine.paired_bootstrap(
        baseline_scores=data["gpt-3.5"].values,
        treatment_scores=data["gpt-4"].values,
        n_resamples=1000,
        ci=0.95,
    )

    # delta = mean(gpt-4) - mean(gpt-3.5) = 0.942 - 0.614 = 0.328
    assert isinstance(result, ComparisonResult)
    assert result.delta_mean == pytest.approx(0.328, rel=1e-2)
    assert result.p_value < 0.05
    assert result.ci_lower > 0.0  # Strict improvement


def test_stats_engine_zero_variance():
    engine = StatsEngine()
    # Arrays are identical, so deltas are all 0
    scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    result = engine.paired_bootstrap(
        baseline_scores=scores, treatment_scores=scores, n_resamples=100
    )

    assert result.delta_mean == 0.0
    assert result.p_value == 1.0
    assert result.ci_lower == 0.0
    assert result.ci_upper == 0.0


def test_stats_engine_adjusts_p_values_for_holm_and_bh():
    engine = StatsEngine()
    p_values = [0.01, 0.03, 0.04]

    assert engine.adjust_p_values(p_values, method="none") == [0.01, 0.03, 0.04]
    assert engine.adjust_p_values(p_values, method="holm") == [0.03, 0.06, 0.06]
    assert engine.adjust_p_values(p_values, method="bh") == [0.03, 0.04, 0.04]
    assert engine.adjust_p_values(p_values, method=PValueCorrection.HOLM) == [
        0.03,
        0.06,
        0.06,
    ]


def test_stats_engine_rejects_unsupported_p_value_adjustment_method():
    engine = StatsEngine()

    with pytest.raises(ValueError, match="Unsupported p-value adjustment method"):
        engine.adjust_p_values([0.01], method="unsupported")
