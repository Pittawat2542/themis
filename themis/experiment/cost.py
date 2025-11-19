"""Cost tracking and estimation for LLM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an experiment.

    Attributes:
        total_cost: Total cost in USD
        generation_cost: Cost of generation API calls
        evaluation_cost: Cost of LLM-based evaluation (if applicable)
        per_sample_costs: List of costs per sample
        per_model_costs: Cost breakdown by model
        token_counts: Token usage statistics
        api_calls: Total number of API calls
        currency: Currency code (default: USD)
    """

    total_cost: float
    generation_cost: float
    evaluation_cost: float = 0.0
    per_sample_costs: list[float] = field(default_factory=list)
    per_model_costs: dict[str, float] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    api_calls: int = 0
    currency: str = "USD"

    def __post_init__(self):
        """Validate cost breakdown."""
        if self.total_cost < 0:
            raise ValueError("Total cost cannot be negative")
        if self.generation_cost < 0:
            raise ValueError("Generation cost cannot be negative")
        if self.evaluation_cost < 0:
            raise ValueError("Evaluation cost cannot be negative")


@dataclass
class CostEstimate:
    """Cost estimate for an experiment.

    Attributes:
        estimated_cost: Expected cost in USD
        lower_bound: Lower bound of 95% confidence interval
        upper_bound: Upper bound of 95% confidence interval
        breakdown_by_phase: Cost breakdown by experiment phase
        assumptions: Assumptions used for estimation
        currency: Currency code (default: USD)
    """

    estimated_cost: float
    lower_bound: float
    upper_bound: float
    breakdown_by_phase: dict[str, float] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    currency: str = "USD"


class CostTracker:
    """Tracks costs during experiment execution.

    This class accumulates costs from generation and evaluation steps,
    providing detailed breakdowns and per-sample tracking.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record_generation("gpt-4", 100, 50, 0.0045)
        >>> tracker.record_generation("gpt-4", 120, 60, 0.0054)
        >>> breakdown = tracker.get_breakdown()
        >>> print(f"Total cost: ${breakdown.total_cost:.4f}")
    """

    def __init__(self):
        """Initialize cost tracker."""
        self._generation_costs: list[tuple[str, float]] = []
        self._evaluation_costs: list[tuple[str, float]] = []
        self._token_counts: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._per_model_costs: dict[str, float] = {}
        self._per_sample_costs: list[float] = []
        self._api_calls: int = 0

    def record_generation(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        """Record cost of a generation call.

        Args:
            model: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Cost in USD
        """
        self._generation_costs.append((model, cost))
        self._token_counts["prompt_tokens"] += prompt_tokens
        self._token_counts["completion_tokens"] += completion_tokens
        self._token_counts["total_tokens"] += prompt_tokens + completion_tokens
        self._per_model_costs[model] = self._per_model_costs.get(model, 0.0) + cost
        self._per_sample_costs.append(cost)
        self._api_calls += 1

    def record_evaluation(self, metric: str, cost: float) -> None:
        """Record cost of LLM-based evaluation.

        Args:
            metric: Metric name that incurred the cost
            cost: Cost in USD
        """
        self._evaluation_costs.append((metric, cost))
        # Evaluation costs also count as API calls
        self._api_calls += 1

    def get_breakdown(self) -> CostBreakdown:
        """Get detailed cost breakdown.

        Returns:
            CostBreakdown with all accumulated costs
        """
        generation_cost = sum(cost for _, cost in self._generation_costs)
        evaluation_cost = sum(cost for _, cost in self._evaluation_costs)
        total_cost = generation_cost + evaluation_cost

        return CostBreakdown(
            total_cost=total_cost,
            generation_cost=generation_cost,
            evaluation_cost=evaluation_cost,
            per_sample_costs=self._per_sample_costs.copy(),
            per_model_costs=self._per_model_costs.copy(),
            token_counts=self._token_counts.copy(),
            api_calls=self._api_calls,
        )

    def reset(self) -> None:
        """Reset all tracked costs."""
        self._generation_costs.clear()
        self._evaluation_costs.clear()
        self._token_counts = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._per_model_costs.clear()
        self._per_sample_costs.clear()
        self._api_calls = 0


class BudgetMonitor:
    """Monitor and enforce budget limits during experiments.

    Example:
        >>> monitor = BudgetMonitor(max_cost=10.0, alert_threshold=0.8)
        >>> monitor.add_cost(7.0)
        >>> within_budget, message = monitor.check_budget()
        >>> print(message)  # "Warning: 70% of budget used"
        >>> monitor.add_cost(4.0)  # Exceeds budget
        >>> within_budget, message = monitor.check_budget()
        >>> print(message)  # "Budget exceeded: $11.00 >= $10.00"
    """

    def __init__(self, max_cost: float, alert_threshold: float = 0.8):
        """Initialize budget monitor.

        Args:
            max_cost: Maximum allowed cost in USD
            alert_threshold: Threshold (0.0-1.0) for warning alerts

        Raises:
            ValueError: If max_cost is negative or alert_threshold is invalid
        """
        if max_cost < 0:
            raise ValueError("Max cost cannot be negative")
        if not 0.0 <= alert_threshold <= 1.0:
            raise ValueError("Alert threshold must be between 0.0 and 1.0")

        self.max_cost = max_cost
        self.alert_threshold = alert_threshold
        self.current_cost = 0.0

    def add_cost(self, cost: float) -> None:
        """Add cost to current total.

        Args:
            cost: Cost to add in USD
        """
        self.current_cost += cost

    def check_budget(self) -> tuple[bool, str]:
        """Check if budget is within limits.

        Returns:
            Tuple of (within_budget, message)
            - within_budget: True if under max_cost
            - message: Status message or warning
        """
        if self.current_cost >= self.max_cost:
            return (
                False,
                f"Budget exceeded: ${self.current_cost:.2f} >= ${self.max_cost:.2f}",
            )

        if self.current_cost >= self.max_cost * self.alert_threshold:
            percentage = (self.current_cost / self.max_cost) * 100
            return (
                True,
                f"Warning: {percentage:.0f}% of budget used "
                f"(${self.current_cost:.2f} / ${self.max_cost:.2f})",
            )

        return True, "Budget OK"

    def remaining_budget(self) -> float:
        """Get remaining budget.

        Returns:
            Remaining budget in USD (may be negative if exceeded)
        """
        return self.max_cost - self.current_cost

    def percentage_used(self) -> float:
        """Get percentage of budget used.

        Returns:
            Percentage (0.0-100.0+) of budget used
        """
        if self.max_cost == 0:
            return 100.0 if self.current_cost > 0 else 0.0
        return (self.current_cost / self.max_cost) * 100


def estimate_experiment_cost(
    model: str,
    dataset_size: int,
    avg_prompt_tokens: int = 500,
    avg_completion_tokens: int = 300,
    confidence_level: float = 0.95,
) -> CostEstimate:
    """Estimate total cost for an experiment.

    Args:
        model: Model identifier
        dataset_size: Number of samples in dataset
        avg_prompt_tokens: Average prompt tokens per sample
        avg_completion_tokens: Average completion tokens per sample
        confidence_level: Confidence level for bounds (default: 0.95)

    Returns:
        CostEstimate with expected cost and confidence bounds

    Example:
        >>> estimate = estimate_experiment_cost("gpt-4", 100, 500, 300)
        >>> print(f"Estimated cost: ${estimate.estimated_cost:.2f}")
        >>> print(f"Range: ${estimate.lower_bound:.2f} - ${estimate.upper_bound:.2f}")
    """
    from themis.experiment.pricing import calculate_cost

    # Calculate cost per sample
    cost_per_sample = calculate_cost(model, avg_prompt_tokens, avg_completion_tokens)

    # Estimate total cost
    estimated_cost = cost_per_sample * dataset_size

    # Calculate confidence bounds (assuming ~20% variance)
    variance_factor = 0.2
    margin = estimated_cost * variance_factor * (1 - (1 - confidence_level))

    lower_bound = max(0.0, estimated_cost - margin)
    upper_bound = estimated_cost + margin

    breakdown = {
        "generation": estimated_cost,
        "evaluation": 0.0,  # No LLM-based evaluation assumed
    }

    assumptions = {
        "model": model,
        "dataset_size": dataset_size,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "cost_per_sample": cost_per_sample,
        "confidence_level": confidence_level,
    }

    return CostEstimate(
        estimated_cost=estimated_cost,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        breakdown_by_phase=breakdown,
        assumptions=assumptions,
    )


__all__ = [
    "CostBreakdown",
    "CostEstimate",
    "CostTracker",
    "BudgetMonitor",
    "estimate_experiment_cost",
]
