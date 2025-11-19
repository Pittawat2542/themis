"""Cost estimation and tracking commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from themis.experiment.cost import estimate_experiment_cost
from themis.experiment.pricing import (
    compare_provider_costs,
    get_all_models,
    get_provider_pricing,
)


def estimate_cost_command(
    *,
    model: Annotated[
        str, Parameter(help="Model identifier (e.g., gpt-4, claude-3-5-sonnet)")
    ],
    dataset_size: Annotated[int, Parameter(help="Number of samples in dataset")],
    avg_prompt_tokens: Annotated[
        int, Parameter(help="Average prompt tokens per sample")
    ] = 500,
    avg_completion_tokens: Annotated[
        int, Parameter(help="Average completion tokens per sample")
    ] = 300,
) -> int:
    """Estimate cost for an experiment before running.

    Examples:
        # Estimate cost for 100 samples with GPT-4
        uv run python -m themis.cli estimate-cost \\
          --model gpt-4 \\
          --dataset-size 100

        # Custom token estimates
        uv run python -m themis.cli estimate-cost \\
          --model claude-3-5-sonnet-20241022 \\
          --dataset-size 1000 \\
          --avg-prompt-tokens 800 \\
          --avg-completion-tokens 400
    """
    try:
        estimate = estimate_experiment_cost(
            model=model,
            dataset_size=dataset_size,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
        )

        print("=" * 80)
        print("Cost Estimate")
        print("=" * 80)
        print(f"\nModel: {model}")
        print(f"Dataset size: {dataset_size} samples")
        print(
            f"Avg tokens per sample: {avg_prompt_tokens} prompt + {avg_completion_tokens} completion"
        )

        print("\n💰 Estimated Cost")
        print(f"  Total: ${estimate.estimated_cost:.4f}")
        print(f"  Per sample: ${estimate.assumptions['cost_per_sample']:.6f}")
        print(f"  95% CI: ${estimate.lower_bound:.4f} - ${estimate.upper_bound:.4f}")

        print("\n📊 Breakdown")
        for phase, cost in estimate.breakdown_by_phase.items():
            print(f"  {phase.capitalize()}: ${cost:.4f}")

        print("\n" + "=" * 80)

        # Warning if cost is high
        if estimate.estimated_cost > 10.0:
            print(
                f"\n⚠️  Warning: Estimated cost is ${estimate.estimated_cost:.2f}. "
                "Consider using --limit for initial testing."
            )

        return 0

    except Exception as e:
        print(f"Error estimating cost: {e}")
        return 1


def show_pricing_command(
    *,
    model: Annotated[
        str | None, Parameter(help="Show pricing for specific model")
    ] = None,
    list_all: Annotated[bool, Parameter(help="List all available models")] = False,
    compare_models: Annotated[
        list[str] | None, Parameter(help="Compare costs for multiple models")
    ] = None,
) -> int:
    """Show pricing information for LLM models.

    Examples:
        # Show pricing for a specific model
        uv run python -m themis.cli show-pricing --model gpt-4

        # List all models with pricing
        uv run python -m themis.cli show-pricing --list-all

        # Compare pricing across models (use repeated --compare-models flags)
        uv run python -m themis.cli show-pricing \\
          --compare-models gpt-4 \\
          --compare-models gpt-3.5-turbo \\
          --compare-models claude-3-haiku-20240307
    """
    try:
        if list_all:
            models = get_all_models()
            print("=" * 80)
            print(f"Available Models ({len(models)} total)")
            print("=" * 80)
            print("\nModel pricing (per 1M tokens):\n")

            for model_name in sorted(models):
                pricing = get_provider_pricing(model_name)
                prompt_price = pricing["prompt_tokens"] * 1_000_000
                completion_price = pricing["completion_tokens"] * 1_000_000
                print(
                    f"  {model_name:40s} | "
                    f"Prompt: ${prompt_price:6.2f} | "
                    f"Completion: ${completion_price:6.2f}"
                )

            print("\n" + "=" * 80)
            return 0

        if compare_models:
            # Compare costs for standard workload
            prompt_tokens = 1000
            completion_tokens = 500

            costs = compare_provider_costs(
                prompt_tokens, completion_tokens, compare_models
            )

            print("=" * 80)
            print(
                f"Cost Comparison ({prompt_tokens} prompt + {completion_tokens} completion tokens)"
            )
            print("=" * 80)
            print()

            # Sort by cost
            sorted_costs = sorted(costs.items(), key=lambda x: x[1])

            for model_name, cost in sorted_costs:
                # Calculate cost per 1M tokens for comparison
                pricing = get_provider_pricing(model_name)
                prompt_price = pricing["prompt_tokens"] * 1_000_000
                completion_price = pricing["completion_tokens"] * 1_000_000

                print(f"  {model_name:40s} | ${cost:.6f}")
                print(
                    f"    {'':40s} | (${prompt_price:.2f} / ${completion_price:.2f} per 1M)"
                )

            # Show relative costs
            if sorted_costs:
                cheapest_cost = sorted_costs[0][1]
                print(f"\nRelative costs (vs {sorted_costs[0][0]}):")
                for model_name, cost in sorted_costs[1:]:
                    multiplier = cost / cheapest_cost if cheapest_cost > 0 else 0
                    print(f"  {model_name:40s} | {multiplier:.1f}x more expensive")

            print("\n" + "=" * 80)
            return 0

        if model:
            pricing = get_provider_pricing(model)
            prompt_price = pricing["prompt_tokens"] * 1_000_000
            completion_price = pricing["completion_tokens"] * 1_000_000

            print("=" * 80)
            print(f"Pricing for {model}")
            print("=" * 80)
            print(f"\nPrompt tokens: ${prompt_price:.2f} per 1M tokens")
            print(f"Completion tokens: ${completion_price:.2f} per 1M tokens")

            # Show example costs
            print("\nExample costs:")
            examples = [
                (100, 50, "Short query"),
                (500, 300, "Medium query"),
                (1000, 500, "Long query"),
            ]

            for prompt_tok, completion_tok, label in examples:
                from themis.experiment.pricing import calculate_cost

                cost = calculate_cost(model, prompt_tok, completion_tok)
                print(
                    f"  {label:15s} ({prompt_tok:4d} + {completion_tok:4d} tokens): ${cost:.6f}"
                )

            print("\n" + "=" * 80)
            return 0

        # No options provided
        print("Error: Must specify --model, --list-all, or --compare-models")
        print("Use --help for usage information")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


__all__ = ["estimate_cost_command", "show_pricing_command"]
