"""Cost tracking utilities for monitoring LLM API usage and costs.

This module provides tools to track token usage, API costs, and generate
cost reports across experiments and providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from themis.core import entities as core_entities

# Provider pricing per 1M tokens (as of 2024)
# Format: {provider_model: (input_cost_per_1m, output_cost_per_1m)}
DEFAULT_PRICING = {
    # OpenAI GPT-4
    "gpt-4": (30.0, 60.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    # OpenAI GPT-3.5
    "gpt-3.5-turbo": (0.5, 1.5),
    # Anthropic Claude
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
    # Meta Llama (via cloud providers - approximate)
    "llama-3-70b": (0.9, 0.9),
    "llama-3-8b": (0.2, 0.2),
    # Fake/local models
    "fake": (0.0, 0.0),
}


@dataclass
class TokenUsage:
    """Token usage statistics for a single API call.

    Attributes:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        total_tokens: Total tokens (input + output)
    """

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostRecord:
    """Cost record for a single generation.

    Attributes:
        model_identifier: Model name/identifier
        provider: Provider name
        usage: Token usage statistics
        input_cost: Cost for input tokens (in USD)
        output_cost: Cost for output tokens (in USD)
        total_cost: Total cost (in USD)
        metadata: Additional metadata (e.g., timestamp, run_id)
    """

    model_identifier: str
    provider: str
    usage: TokenUsage
    input_cost: float
    output_cost: float
    total_cost: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class CostSummary:
    """Aggregated cost summary across multiple generations.

    Attributes:
        total_cost: Total cost in USD
        total_tokens: Total number of tokens
        total_input_tokens: Total input tokens
        total_output_tokens: Total output tokens
        num_requests: Number of API requests
        cost_by_model: Cost breakdown by model
        cost_by_provider: Cost breakdown by provider
    """

    total_cost: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    num_requests: int
    cost_by_model: Dict[str, float]
    cost_by_provider: Dict[str, float]


class CostTracker:
    """Track and compute costs for LLM API usage.

    This class maintains a record of all API calls and their costs,
    with support for custom pricing models and cost aggregation.
    """

    def __init__(
        self,
        pricing: Dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize cost tracker.

        Args:
            pricing: Custom pricing dictionary mapping model names to
                (input_cost_per_1m, output_cost_per_1m) tuples.
                Defaults to DEFAULT_PRICING if not provided.
        """
        self.pricing = pricing or DEFAULT_PRICING.copy()
        self.records: List[CostRecord] = []

    def add_pricing(
        self,
        model: str,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
    ) -> None:
        """Add or update pricing for a model.

        Args:
            model: Model identifier
            input_cost_per_1m: Cost per 1M input tokens in USD
            output_cost_per_1m: Cost per 1M output tokens in USD
        """
        self.pricing[model] = (input_cost_per_1m, output_cost_per_1m)

    def track_generation(
        self,
        record: core_entities.GenerationRecord,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> CostRecord:
        """Track cost for a generation record.

        Args:
            record: Generation record to track
            input_tokens: Number of input tokens (if None, estimated from prompt)
            output_tokens: Number of output tokens (if None, estimated from output)

        Returns:
            CostRecord with computed costs
        """
        model_id = record.task.model.identifier
        provider = record.task.model.provider

        # Extract or estimate token counts
        if input_tokens is None:
            input_tokens = self._estimate_tokens(record.task.prompt.text)

        if output_tokens is None and record.output:
            output_tokens = self._estimate_tokens(record.output.text)
        elif output_tokens is None:
            output_tokens = 0

        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # Compute costs
        input_cost, output_cost = self._compute_cost(model_id, usage)
        total_cost = input_cost + output_cost

        cost_record = CostRecord(
            model_identifier=model_id,
            provider=provider,
            usage=usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            metadata={
                "sample_id": record.task.metadata.get("sample_id"),
                "run_id": record.task.metadata.get("run_id"),
            },
        )

        self.records.append(cost_record)
        return cost_record

    def get_summary(self) -> CostSummary:
        """Compute aggregated cost summary across all tracked records.

        Returns:
            CostSummary with aggregated statistics
        """
        if not self.records:
            return CostSummary(
                total_cost=0.0,
                total_tokens=0,
                total_input_tokens=0,
                total_output_tokens=0,
                num_requests=0,
                cost_by_model={},
                cost_by_provider={},
            )

        total_cost = sum(r.total_cost for r in self.records)
        total_input_tokens = sum(r.usage.input_tokens for r in self.records)
        total_output_tokens = sum(r.usage.output_tokens for r in self.records)

        # Aggregate by model
        cost_by_model: Dict[str, float] = {}
        for record in self.records:
            model = record.model_identifier
            cost_by_model[model] = cost_by_model.get(model, 0.0) + record.total_cost

        # Aggregate by provider
        cost_by_provider: Dict[str, float] = {}
        for record in self.records:
            provider = record.provider
            cost_by_provider[provider] = (
                cost_by_provider.get(provider, 0.0) + record.total_cost
            )

        return CostSummary(
            total_cost=total_cost,
            total_tokens=total_input_tokens + total_output_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            num_requests=len(self.records),
            cost_by_model=cost_by_model,
            cost_by_provider=cost_by_provider,
        )

    def export_records(self, path: str | Path) -> None:
        """Export cost records to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "records": [
                {
                    "model": r.model_identifier,
                    "provider": r.provider,
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                    "total_tokens": r.usage.total_tokens,
                    "input_cost": r.input_cost,
                    "output_cost": r.output_cost,
                    "total_cost": r.total_cost,
                    "metadata": r.metadata,
                }
                for r in self.records
            ],
            "summary": {
                "total_cost": self.get_summary().total_cost,
                "total_tokens": self.get_summary().total_tokens,
                "num_requests": len(self.records),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_cost(
        self,
        model: str,
        usage: TokenUsage,
    ) -> tuple[float, float]:
        """Compute input and output costs for a model.

        Args:
            model: Model identifier
            usage: Token usage statistics

        Returns:
            Tuple of (input_cost, output_cost) in USD
        """
        # Try exact match first
        pricing = self.pricing.get(model)

        # If no exact match, try prefix matching
        if pricing is None:
            for price_key in self.pricing:
                if model.startswith(price_key):
                    pricing = self.pricing[price_key]
                    break

        # Fall back to generic pricing if model not found
        if pricing is None:
            # Use a reasonable default ($1 per 1M tokens)
            pricing = (1.0, 1.0)

        input_cost_per_1m, output_cost_per_1m = pricing

        input_cost = (usage.input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (usage.output_tokens / 1_000_000) * output_cost_per_1m

        return input_cost, output_cost

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate of token count from text.

        Uses a simple heuristic: ~4 characters per token on average.
        For accurate counts, use provider-specific tokenizers.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)


def format_cost_summary(summary: CostSummary) -> str:
    """Format cost summary as human-readable string.

    Args:
        summary: Cost summary to format

    Returns:
        Formatted string representation
    """
    lines = [
        "Cost Summary",
        "=" * 50,
        f"Total Cost:        ${summary.total_cost:.4f}",
        f"Total Tokens:      {summary.total_tokens:,}",
        f"  Input Tokens:    {summary.total_input_tokens:,}",
        f"  Output Tokens:   {summary.total_output_tokens:,}",
        f"API Requests:      {summary.num_requests:,}",
        "",
    ]

    if summary.cost_by_model:
        lines.append("Cost by Model:")
        lines.append("-" * 50)
        for model, cost in sorted(
            summary.cost_by_model.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = (cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            lines.append(f"  {model:30s} ${cost:8.4f} ({pct:5.1f}%)")
        lines.append("")

    if summary.cost_by_provider:
        lines.append("Cost by Provider:")
        lines.append("-" * 50)
        for provider, cost in sorted(
            summary.cost_by_provider.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = (cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            lines.append(f"  {provider:30s} ${cost:8.4f} ({pct:5.1f}%)")

    return "\n".join(lines)


__all__ = [
    "TokenUsage",
    "CostRecord",
    "CostSummary",
    "CostTracker",
    "DEFAULT_PRICING",
    "format_cost_summary",
]
