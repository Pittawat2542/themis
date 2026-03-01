"""Data entities for multi-experiment comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComparisonRow:
    """Single experiment in a multi-experiment comparison."""

    run_id: str
    metric_values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    sample_count: int = 0
    failure_count: int = 0

    def get_metric(self, metric_name: str) -> float | None:
        """Get metric value by name.

        Special metric names:
        - 'cost' or 'total_cost': Checks metadata first, then metric_values
        - Any other name: Returns from metric_values dict
        """
        # Handle special cost metrics - check metadata first
        if metric_name in ("cost", "total_cost"):
            cost_data = self.metadata.get("cost")
            if cost_data and "total_cost" in cost_data:
                return cost_data["total_cost"]
            return self.metric_values.get(metric_name)

        return self.metric_values.get(metric_name)

    def get_cost(self) -> float | None:
        """Get total cost if available.

        Returns:
            Total cost in USD, or None if not tracked
        """
        return self.get_metric("cost")


@dataclass
class ConfigDiff:
    """Differences between two experiment configurations."""

    run_id_a: str
    run_id_b: str
    changed_fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    added_fields: dict[str, Any] = field(default_factory=dict)
    removed_fields: dict[str, Any] = field(default_factory=dict)

    def has_differences(self) -> bool:
        """Check if there are any differences."""
        return bool(self.changed_fields or self.added_fields or self.removed_fields)
