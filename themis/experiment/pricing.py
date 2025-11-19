"""Provider pricing database and cost calculation utilities."""

from __future__ import annotations

from typing import Any

# Pricing table for common LLM providers (prices per token in USD)
# Updated as of November 2024
PRICING_TABLE: dict[str, dict[str, float]] = {
    # OpenAI models
    "gpt-4": {
        "prompt_tokens": 0.00003,  # $30 per 1M tokens
        "completion_tokens": 0.00006,  # $60 per 1M tokens
    },
    "gpt-4-32k": {
        "prompt_tokens": 0.00006,
        "completion_tokens": 0.00012,
    },
    "gpt-4-turbo": {
        "prompt_tokens": 0.00001,  # $10 per 1M tokens
        "completion_tokens": 0.00003,  # $30 per 1M tokens
    },
    "gpt-4-turbo-preview": {
        "prompt_tokens": 0.00001,
        "completion_tokens": 0.00003,
    },
    "gpt-3.5-turbo": {
        "prompt_tokens": 0.0000005,  # $0.50 per 1M tokens
        "completion_tokens": 0.0000015,  # $1.50 per 1M tokens
    },
    "gpt-3.5-turbo-16k": {
        "prompt_tokens": 0.000003,
        "completion_tokens": 0.000004,
    },
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": {
        "prompt_tokens": 0.000003,  # $3 per 1M tokens
        "completion_tokens": 0.000015,  # $15 per 1M tokens
    },
    "claude-3-opus-20240229": {
        "prompt_tokens": 0.000015,  # $15 per 1M tokens
        "completion_tokens": 0.000075,  # $75 per 1M tokens
    },
    "claude-3-sonnet-20240229": {
        "prompt_tokens": 0.000003,
        "completion_tokens": 0.000015,
    },
    "claude-3-haiku-20240307": {
        "prompt_tokens": 0.00000025,  # $0.25 per 1M tokens
        "completion_tokens": 0.00000125,  # $1.25 per 1M tokens
    },
    # Google models
    "gemini-pro": {
        "prompt_tokens": 0.00000025,
        "completion_tokens": 0.0000005,
    },
    "gemini-1.5-pro": {
        "prompt_tokens": 0.00000125,  # $1.25 per 1M tokens
        "completion_tokens": 0.000005,  # $5 per 1M tokens
    },
    "gemini-1.5-flash": {
        "prompt_tokens": 0.000000075,  # $0.075 per 1M tokens
        "completion_tokens": 0.0000003,  # $0.30 per 1M tokens
    },
    # Mistral models
    "mistral-large-latest": {
        "prompt_tokens": 0.000002,  # $2 per 1M tokens
        "completion_tokens": 0.000006,  # $6 per 1M tokens
    },
    "mistral-medium-latest": {
        "prompt_tokens": 0.0000027,
        "completion_tokens": 0.0000081,
    },
    "mistral-small-latest": {
        "prompt_tokens": 0.000001,
        "completion_tokens": 0.000003,
    },
    # Cohere models
    "command-r-plus": {
        "prompt_tokens": 0.000003,
        "completion_tokens": 0.000015,
    },
    "command-r": {
        "prompt_tokens": 0.0000005,
        "completion_tokens": 0.0000015,
    },
    # Meta Llama (via various providers - using typical cloud pricing)
    "llama-3.1-70b": {
        "prompt_tokens": 0.00000088,
        "completion_tokens": 0.00000088,
    },
    "llama-3.1-8b": {
        "prompt_tokens": 0.0000002,
        "completion_tokens": 0.0000002,
    },
    # Default fallback for unknown models
    "default": {
        "prompt_tokens": 0.000001,
        "completion_tokens": 0.000002,
    },
}

# Model aliases and variations
MODEL_ALIASES: dict[str, str] = {
    # OpenAI aliases
    "gpt-4-0613": "gpt-4",
    "gpt-4-0314": "gpt-4",
    "gpt-4-1106-preview": "gpt-4-turbo-preview",
    "gpt-4-0125-preview": "gpt-4-turbo-preview",
    "gpt-3.5-turbo-0613": "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301": "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    # Anthropic aliases
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    # Google aliases
    "gemini-pro-1.0": "gemini-pro",
    "gemini-1.5-pro-latest": "gemini-1.5-pro",
    "gemini-1.5-flash-latest": "gemini-1.5-flash",
}


def normalize_model_name(model: str) -> str:
    """Normalize model name to canonical form.

    Args:
        model: Model identifier (may include provider prefix)

    Returns:
        Normalized model name

    Example:
        >>> normalize_model_name("openai/gpt-4-0613")
        'gpt-4'
        >>> normalize_model_name("claude-3-opus")
        'claude-3-opus-20240229'
    """
    # Remove provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
    if "/" in model:
        model = model.split("/", 1)[1]

    # Look up alias
    model = MODEL_ALIASES.get(model, model)

    return model


def get_provider_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model.

    Args:
        model: Model identifier

    Returns:
        Dict with 'prompt_tokens' and 'completion_tokens' prices per token

    Example:
        >>> pricing = get_provider_pricing("gpt-4")
        >>> print(f"Prompt: ${pricing['prompt_tokens'] * 1_000_000:.2f}/1M tokens")
        Prompt: $30.00/1M tokens
    """
    normalized = normalize_model_name(model)

    # Check if we have pricing for this model
    if normalized in PRICING_TABLE:
        return PRICING_TABLE[normalized].copy()

    # Try to find a partial match (e.g., "gpt-4-turbo-2024-04-09" matches "gpt-4-turbo")
    for known_model in PRICING_TABLE:
        if known_model in normalized or normalized.startswith(known_model):
            return PRICING_TABLE[known_model].copy()

    # Fallback to default pricing
    return PRICING_TABLE["default"].copy()


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, float] | None = None,
) -> float:
    """Calculate cost for a model completion.

    Args:
        model: Model identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        pricing: Optional custom pricing (if None, uses default pricing table)

    Returns:
        Total cost in USD

    Example:
        >>> cost = calculate_cost("gpt-4", 1000, 500)
        >>> print(f"Cost: ${cost:.4f}")
        Cost: $0.0600
    """
    if pricing is None:
        pricing = get_provider_pricing(model)

    prompt_cost = prompt_tokens * pricing["prompt_tokens"]
    completion_cost = completion_tokens * pricing["completion_tokens"]

    return prompt_cost + completion_cost


def compare_provider_costs(
    prompt_tokens: int,
    completion_tokens: int,
    models: list[str],
) -> dict[str, float]:
    """Compare costs across multiple providers for same workload.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        models: List of model identifiers to compare

    Returns:
        Dict mapping model names to costs

    Example:
        >>> costs = compare_provider_costs(
        ...     1000, 500, ["gpt-4", "gpt-3.5-turbo", "claude-3-haiku"]
        ... )
        >>> for model, cost in sorted(costs.items(), key=lambda x: x[1]):
        ...     print(f"{model}: ${cost:.4f}")
        claude-3-haiku: $0.0009
        gpt-3.5-turbo: $0.0013
        gpt-4: $0.0600
    """
    costs = {}
    for model in models:
        costs[model] = calculate_cost(model, prompt_tokens, completion_tokens)
    return costs


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate number of tokens from text.

    This is a rough approximation. For accurate token counts,
    use the model's tokenizer.

    Args:
        text: Input text
        chars_per_token: Average characters per token (default: 4.0)

    Returns:
        Estimated token count

    Example:
        >>> text = "This is a sample text for token estimation."
        >>> tokens = estimate_tokens(text)
        >>> print(f"Estimated tokens: {tokens}")
        Estimated tokens: 11
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def get_all_models() -> list[str]:
    """Get list of all models with known pricing.

    Returns:
        List of model identifiers
    """
    return [k for k in PRICING_TABLE.keys() if k != "default"]


def get_pricing_summary() -> dict[str, Any]:
    """Get summary of pricing for all models.

    Returns:
        Dict with model pricing information

    Example:
        >>> summary = get_pricing_summary()
        >>> print(f"Total models: {summary['total_models']}")
        >>> print(f"Cheapest: {summary['cheapest_model']}")
    """
    models = get_all_models()

    # Find cheapest and most expensive (based on prompt + completion average)
    model_avg_costs = {}
    for model in models:
        pricing = PRICING_TABLE[model]
        avg_cost = (pricing["prompt_tokens"] + pricing["completion_tokens"]) / 2
        model_avg_costs[model] = avg_cost

    cheapest = min(model_avg_costs.items(), key=lambda x: x[1])
    most_expensive = max(model_avg_costs.items(), key=lambda x: x[1])

    return {
        "total_models": len(models),
        "cheapest_model": cheapest[0],
        "cheapest_avg_cost_per_token": cheapest[1],
        "most_expensive_model": most_expensive[0],
        "most_expensive_avg_cost_per_token": most_expensive[1],
        "models": models,
    }


__all__ = [
    "PRICING_TABLE",
    "MODEL_ALIASES",
    "normalize_model_name",
    "get_provider_pricing",
    "calculate_cost",
    "compare_provider_costs",
    "estimate_tokens",
    "get_all_models",
    "get_pricing_summary",
]
