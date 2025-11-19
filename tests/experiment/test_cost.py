"""Tests for cost tracking functionality."""

import pytest

from themis.experiment.cost import BudgetMonitor, CostBreakdown, CostTracker


def test_cost_tracker_initialization():
    """Test CostTracker initializes with empty state."""
    tracker = CostTracker()
    breakdown = tracker.get_breakdown()

    assert breakdown.total_cost == 0.0
    assert breakdown.generation_cost == 0.0
    assert breakdown.evaluation_cost == 0.0
    assert breakdown.api_calls == 0
    assert len(breakdown.per_sample_costs) == 0
    assert len(breakdown.per_model_costs) == 0


def test_cost_tracker_record_generation():
    """Test recording generation costs."""
    tracker = CostTracker()

    tracker.record_generation(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0045,
    )

    breakdown = tracker.get_breakdown()

    assert breakdown.generation_cost == 0.0045
    assert breakdown.total_cost == 0.0045
    assert breakdown.token_counts["prompt_tokens"] == 100
    assert breakdown.token_counts["completion_tokens"] == 50
    assert breakdown.token_counts["total_tokens"] == 150
    assert breakdown.api_calls == 1
    assert len(breakdown.per_sample_costs) == 1
    assert breakdown.per_sample_costs[0] == 0.0045
    assert breakdown.per_model_costs["gpt-4"] == 0.0045


def test_cost_tracker_multiple_generations():
    """Test accumulating costs across multiple generations."""
    tracker = CostTracker()

    tracker.record_generation("gpt-4", 100, 50, 0.0045)
    tracker.record_generation("gpt-4", 120, 60, 0.0054)
    tracker.record_generation("gpt-3.5-turbo", 100, 50, 0.0002)

    breakdown = tracker.get_breakdown()

    assert breakdown.generation_cost == pytest.approx(0.0101, abs=0.0001)
    assert breakdown.token_counts["prompt_tokens"] == 320
    assert breakdown.token_counts["completion_tokens"] == 160
    assert breakdown.api_calls == 3
    assert len(breakdown.per_sample_costs) == 3
    assert breakdown.per_model_costs["gpt-4"] == pytest.approx(0.0099, abs=0.0001)
    assert breakdown.per_model_costs["gpt-3.5-turbo"] == pytest.approx(
        0.0002, abs=0.0001
    )


def test_cost_tracker_record_evaluation():
    """Test recording evaluation costs."""
    tracker = CostTracker()

    tracker.record_evaluation("rubric_judge", 0.005)

    breakdown = tracker.get_breakdown()

    assert breakdown.evaluation_cost == 0.005
    assert breakdown.total_cost == 0.005
    assert breakdown.api_calls == 1


def test_cost_tracker_combined_costs():
    """Test tracking both generation and evaluation costs."""
    tracker = CostTracker()

    tracker.record_generation("gpt-4", 100, 50, 0.0045)
    tracker.record_evaluation("rubric_judge", 0.005)
    tracker.record_generation("gpt-3.5-turbo", 100, 50, 0.0002)

    breakdown = tracker.get_breakdown()

    assert breakdown.generation_cost == pytest.approx(0.0047, abs=0.0001)
    assert breakdown.evaluation_cost == 0.005
    assert breakdown.total_cost == pytest.approx(0.0097, abs=0.0001)
    assert breakdown.api_calls == 3


def test_cost_tracker_reset():
    """Test resetting cost tracker."""
    tracker = CostTracker()

    tracker.record_generation("gpt-4", 100, 50, 0.0045)
    tracker.record_evaluation("rubric_judge", 0.005)

    tracker.reset()

    breakdown = tracker.get_breakdown()

    assert breakdown.total_cost == 0.0
    assert breakdown.generation_cost == 0.0
    assert breakdown.evaluation_cost == 0.0
    assert breakdown.api_calls == 0
    assert len(breakdown.per_sample_costs) == 0
    assert len(breakdown.per_model_costs) == 0


def test_cost_breakdown_validation():
    """Test CostBreakdown validates inputs."""
    # Valid breakdown
    breakdown = CostBreakdown(total_cost=1.0, generation_cost=0.8, evaluation_cost=0.2)
    assert breakdown.total_cost == 1.0

    # Negative total cost
    with pytest.raises(ValueError, match="Total cost cannot be negative"):
        CostBreakdown(total_cost=-1.0, generation_cost=0.0)

    # Negative generation cost
    with pytest.raises(ValueError, match="Generation cost cannot be negative"):
        CostBreakdown(total_cost=1.0, generation_cost=-0.5)

    # Negative evaluation cost
    with pytest.raises(ValueError, match="Evaluation cost cannot be negative"):
        CostBreakdown(total_cost=1.0, generation_cost=0.5, evaluation_cost=-0.5)


def test_budget_monitor_initialization():
    """Test BudgetMonitor initializes correctly."""
    monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)

    assert monitor.max_cost == 10.0
    assert monitor.alert_threshold == 0.8
    assert monitor.current_cost == 0.0


def test_budget_monitor_invalid_initialization():
    """Test BudgetMonitor rejects invalid parameters."""
    # Negative max cost
    with pytest.raises(ValueError, match="Max cost cannot be negative"):
        BudgetMonitor(max_cost=-10.0)

    # Invalid alert threshold (> 1.0)
    with pytest.raises(ValueError, match="Alert threshold must be between"):
        BudgetMonitor(max_cost=10.0, alert_threshold=1.5)

    # Invalid alert threshold (< 0.0)
    with pytest.raises(ValueError, match="Alert threshold must be between"):
        BudgetMonitor(max_cost=10.0, alert_threshold=-0.1)


def test_budget_monitor_add_cost():
    """Test adding costs to budget monitor."""
    monitor = BudgetMonitor(max_cost=10.0)

    monitor.add_cost(3.5)
    assert monitor.current_cost == 3.5

    monitor.add_cost(2.0)
    assert monitor.current_cost == 5.5


def test_budget_monitor_check_budget_ok():
    """Test budget check when under threshold."""
    monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)

    monitor.add_cost(5.0)
    within_budget, message = monitor.check_budget()

    assert within_budget is True
    assert message == "Budget OK"


def test_budget_monitor_check_budget_warning():
    """Test budget check when over alert threshold."""
    monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)

    monitor.add_cost(8.5)
    within_budget, message = monitor.check_budget()

    assert within_budget is True
    assert "Warning" in message
    assert "85%" in message


def test_budget_monitor_check_budget_exceeded():
    """Test budget check when exceeded."""
    monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)

    monitor.add_cost(11.0)
    within_budget, message = monitor.check_budget()

    assert within_budget is False
    assert "Budget exceeded" in message
    assert "$11.00" in message
    assert "$10.00" in message


def test_budget_monitor_remaining_budget():
    """Test calculating remaining budget."""
    monitor = BudgetMonitor(max_cost=10.0)

    assert monitor.remaining_budget() == 10.0

    monitor.add_cost(3.5)
    assert monitor.remaining_budget() == 6.5

    monitor.add_cost(8.0)
    assert monitor.remaining_budget() == -1.5  # Can go negative


def test_budget_monitor_percentage_used():
    """Test calculating percentage of budget used."""
    monitor = BudgetMonitor(max_cost=10.0)

    assert monitor.percentage_used() == 0.0

    monitor.add_cost(5.0)
    assert monitor.percentage_used() == 50.0

    monitor.add_cost(2.5)
    assert monitor.percentage_used() == 75.0

    monitor.add_cost(5.0)
    assert monitor.percentage_used() == 125.0  # Can exceed 100%


def test_budget_monitor_zero_max_cost():
    """Test budget monitor with zero max cost."""
    monitor = BudgetMonitor(max_cost=0.0)

    # Zero cost should report 0% used
    assert monitor.percentage_used() == 0.0

    # Any cost should report 100% used
    monitor.add_cost(0.01)
    assert monitor.percentage_used() == 100.0


def test_cost_breakdown_defaults():
    """Test CostBreakdown uses correct defaults."""
    breakdown = CostBreakdown(total_cost=1.0, generation_cost=0.8)

    assert breakdown.evaluation_cost == 0.0
    assert breakdown.per_sample_costs == []
    assert breakdown.per_model_costs == {}
    assert breakdown.token_counts == {}
    assert breakdown.api_calls == 0
    assert breakdown.currency == "USD"


def test_cost_tracker_per_sample_costs():
    """Test per-sample cost tracking."""
    tracker = CostTracker()

    costs = [0.0045, 0.0054, 0.0002, 0.0038]
    for cost in costs:
        tracker.record_generation("gpt-4", 100, 50, cost)

    breakdown = tracker.get_breakdown()

    assert len(breakdown.per_sample_costs) == 4
    assert breakdown.per_sample_costs == costs


def test_cost_tracker_per_model_aggregation():
    """Test costs are correctly aggregated per model."""
    tracker = CostTracker()

    # Multiple calls to same model
    tracker.record_generation("gpt-4", 100, 50, 0.0045)
    tracker.record_generation("gpt-4", 100, 50, 0.0045)

    # Different model
    tracker.record_generation("gpt-3.5-turbo", 100, 50, 0.0002)

    breakdown = tracker.get_breakdown()

    assert breakdown.per_model_costs["gpt-4"] == pytest.approx(0.0090, abs=0.0001)
    assert breakdown.per_model_costs["gpt-3.5-turbo"] == pytest.approx(
        0.0002, abs=0.0001
    )


def test_budget_monitor_exact_threshold():
    """Test budget monitor behavior at exact threshold."""
    monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)

    # Exactly at threshold should trigger warning
    monitor.add_cost(8.0)
    within_budget, message = monitor.check_budget()

    assert within_budget is True
    assert "Warning" in message

    # Exactly at max should trigger exceeded
    monitor2 = BudgetMonitor(max_cost=10.0)
    monitor2.add_cost(10.0)
    within_budget, message = monitor2.check_budget()

    assert within_budget is False
    assert "Budget exceeded" in message
