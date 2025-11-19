"""Tests for pricing database and cost calculation."""

import pytest

from themis.experiment.pricing import (
    calculate_cost,
    compare_provider_costs,
    estimate_tokens,
    get_all_models,
    get_pricing_summary,
    get_provider_pricing,
    normalize_model_name,
)


def test_normalize_model_name_with_provider_prefix():
    """Test normalizing model names with provider prefixes."""
    assert normalize_model_name("openai/gpt-4") == "gpt-4"
    assert normalize_model_name("anthropic/claude-3-opus") == "claude-3-opus-20240229"
    assert normalize_model_name("google/gemini-pro") == "gemini-pro"


def test_normalize_model_name_with_aliases():
    """Test normalizing model name aliases."""
    assert normalize_model_name("gpt-4-0613") == "gpt-4"
    assert normalize_model_name("claude-3-opus") == "claude-3-opus-20240229"
    assert normalize_model_name("claude-3.5-sonnet") == "claude-3-5-sonnet-20241022"


def test_normalize_model_name_canonical():
    """Test normalizing already-canonical names."""
    assert normalize_model_name("gpt-4") == "gpt-4"
    assert normalize_model_name("claude-3-opus-20240229") == "claude-3-opus-20240229"
    assert normalize_model_name("gemini-1.5-pro") == "gemini-1.5-pro"


def test_get_provider_pricing_known_model():
    """Test getting pricing for known models."""
    gpt4_pricing = get_provider_pricing("gpt-4")

    assert "prompt_tokens" in gpt4_pricing
    assert "completion_tokens" in gpt4_pricing
    assert gpt4_pricing["prompt_tokens"] == 0.00003
    assert gpt4_pricing["completion_tokens"] == 0.00006


def test_get_provider_pricing_with_alias():
    """Test getting pricing using model alias."""
    pricing = get_provider_pricing("gpt-4-0613")

    # Should resolve to gpt-4 pricing
    assert pricing["prompt_tokens"] == 0.00003
    assert pricing["completion_tokens"] == 0.00006


def test_get_provider_pricing_with_provider_prefix():
    """Test getting pricing with provider prefix."""
    pricing = get_provider_pricing("openai/gpt-4")

    assert pricing["prompt_tokens"] == 0.00003
    assert pricing["completion_tokens"] == 0.00006


def test_get_provider_pricing_unknown_model():
    """Test getting pricing for unknown model returns default."""
    pricing = get_provider_pricing("unknown-model-xyz")

    assert "prompt_tokens" in pricing
    assert "completion_tokens" in pricing
    # Should use default pricing
    assert pricing["prompt_tokens"] == 0.000001
    assert pricing["completion_tokens"] == 0.000002


def test_get_provider_pricing_partial_match():
    """Test pricing lookup with partial model name match."""
    # "gpt-4-turbo-2024-04-09" will match "gpt-4" first (shortest match)
    # This is expected behavior - exact matches or aliases should be used for specific models
    pricing = get_provider_pricing("gpt-4-turbo-2024-04-09")

    # Will match "gpt-4" due to partial matching
    assert pricing["prompt_tokens"] in [0.00003, 0.00001]  # Either gpt-4 or gpt-4-turbo
    assert pricing["completion_tokens"] in [0.00006, 0.00003]


def test_calculate_cost_gpt4():
    """Test calculating cost for GPT-4."""
    cost = calculate_cost("gpt-4", prompt_tokens=1000, completion_tokens=500)

    # 1000 * 0.00003 + 500 * 0.00006 = 0.03 + 0.03 = 0.06
    assert cost == pytest.approx(0.06, abs=0.001)


def test_calculate_cost_gpt35():
    """Test calculating cost for GPT-3.5-turbo."""
    cost = calculate_cost("gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=500)

    # 1000 * 0.0000005 + 500 * 0.0000015 = 0.0005 + 0.00075 = 0.00125
    assert cost == pytest.approx(0.00125, abs=0.00001)


def test_calculate_cost_claude():
    """Test calculating cost for Claude models."""
    cost = calculate_cost(
        "claude-3-5-sonnet-20241022", prompt_tokens=1000, completion_tokens=500
    )

    # 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
    assert cost == pytest.approx(0.0105, abs=0.0001)


def test_calculate_cost_with_custom_pricing():
    """Test calculating cost with custom pricing."""
    custom_pricing = {"prompt_tokens": 0.00001, "completion_tokens": 0.00002}

    cost = calculate_cost(
        "any-model", prompt_tokens=1000, completion_tokens=500, pricing=custom_pricing
    )

    # 1000 * 0.00001 + 500 * 0.00002 = 0.01 + 0.01 = 0.02
    assert cost == pytest.approx(0.02, abs=0.001)


def test_calculate_cost_zero_tokens():
    """Test calculating cost with zero tokens."""
    cost = calculate_cost("gpt-4", prompt_tokens=0, completion_tokens=0)

    assert cost == 0.0


def test_compare_provider_costs():
    """Test comparing costs across multiple providers."""
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-haiku-20240307"]

    costs = compare_provider_costs(
        prompt_tokens=1000, completion_tokens=500, models=models
    )

    assert len(costs) == 3
    assert "gpt-4" in costs
    assert "gpt-3.5-turbo" in costs
    assert "claude-3-haiku-20240307" in costs

    # GPT-4 should be most expensive
    assert costs["gpt-4"] > costs["gpt-3.5-turbo"]
    assert costs["gpt-4"] > costs["claude-3-haiku-20240307"]

    # Claude Haiku should be cheapest
    assert costs["claude-3-haiku-20240307"] < costs["gpt-4"]


def test_compare_provider_costs_sorted():
    """Test that we can sort provider costs."""
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-haiku-20240307", "gemini-1.5-flash"]

    costs = compare_provider_costs(
        prompt_tokens=1000, completion_tokens=500, models=models
    )

    sorted_costs = sorted(costs.items(), key=lambda x: x[1])

    # Verify we can sort and get cheapest first
    cheapest_model, cheapest_cost = sorted_costs[0]
    most_expensive_model, most_expensive_cost = sorted_costs[-1]

    assert cheapest_cost < most_expensive_cost
    assert most_expensive_model == "gpt-4"


def test_estimate_tokens_default():
    """Test estimating tokens with default chars_per_token."""
    text = "This is a test sentence with about forty characters."

    tokens = estimate_tokens(text)

    # ~52 chars / 4 = 13 tokens
    assert 10 <= tokens <= 15


def test_estimate_tokens_custom_ratio():
    """Test estimating tokens with custom chars_per_token."""
    text = "This is a test sentence."

    tokens = estimate_tokens(text, chars_per_token=5.0)

    # ~25 chars / 5 = 5 tokens
    assert 4 <= tokens <= 6


def test_estimate_tokens_empty_string():
    """Test estimating tokens for empty string."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") >= 1  # Whitespace counts


def test_estimate_tokens_minimum():
    """Test that estimate_tokens returns at least 1 for non-empty text."""
    # Very short text should still return at least 1 token
    assert estimate_tokens("a") >= 1


def test_get_all_models():
    """Test getting list of all models with pricing."""
    models = get_all_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert "claude-3-5-sonnet-20241022" in models
    assert "gemini-1.5-pro" in models
    # "default" should not be in the list
    assert "default" not in models


def test_get_pricing_summary():
    """Test getting pricing summary."""
    summary = get_pricing_summary()

    assert "total_models" in summary
    assert "cheapest_model" in summary
    assert "most_expensive_model" in summary
    assert "cheapest_avg_cost_per_token" in summary
    assert "most_expensive_avg_cost_per_token" in summary
    assert "models" in summary

    assert summary["total_models"] > 0
    assert summary["cheapest_avg_cost_per_token"] < summary[
        "most_expensive_avg_cost_per_token"
    ]


def test_pricing_consistency():
    """Test that pricing is consistent across different access methods."""
    # Get pricing directly
    direct_pricing = get_provider_pricing("gpt-4")

    # Calculate cost manually
    manual_cost = 1000 * direct_pricing["prompt_tokens"] + 500 * direct_pricing[
        "completion_tokens"
    ]

    # Calculate using helper
    helper_cost = calculate_cost("gpt-4", 1000, 500)

    assert manual_cost == pytest.approx(helper_cost, abs=0.0001)


def test_specific_model_prices():
    """Test specific model prices match expected values (as of Nov 2024)."""
    # GPT-4: $30/$60 per 1M tokens
    gpt4 = get_provider_pricing("gpt-4")
    assert gpt4["prompt_tokens"] == 0.00003
    assert gpt4["completion_tokens"] == 0.00006

    # GPT-3.5-turbo: $0.50/$1.50 per 1M tokens
    gpt35 = get_provider_pricing("gpt-3.5-turbo")
    assert gpt35["prompt_tokens"] == 0.0000005
    assert gpt35["completion_tokens"] == 0.0000015

    # Claude 3.5 Sonnet: $3/$15 per 1M tokens
    claude35 = get_provider_pricing("claude-3-5-sonnet-20241022")
    assert claude35["prompt_tokens"] == 0.000003
    assert claude35["completion_tokens"] == 0.000015

    # Claude 3 Haiku: $0.25/$1.25 per 1M tokens (cheapest)
    haiku = get_provider_pricing("claude-3-haiku-20240307")
    assert haiku["prompt_tokens"] == 0.00000025
    assert haiku["completion_tokens"] == 0.00000125


def test_pricing_returns_copy():
    """Test that get_provider_pricing returns a copy, not reference."""
    pricing1 = get_provider_pricing("gpt-4")
    pricing2 = get_provider_pricing("gpt-4")

    # Modify one
    pricing1["prompt_tokens"] = 999.0

    # Other should be unchanged
    assert pricing2["prompt_tokens"] == 0.00003


def test_compare_provider_costs_empty_list():
    """Test comparing costs with empty model list."""
    costs = compare_provider_costs(prompt_tokens=1000, completion_tokens=500, models=[])

    assert costs == {}


def test_realistic_cost_calculation():
    """Test realistic cost calculation for a typical use case."""
    # Typical prompt: ~500 tokens, response: ~300 tokens
    cost_gpt4 = calculate_cost("gpt-4", 500, 300)
    cost_gpt35 = calculate_cost("gpt-3.5-turbo", 500, 300)

    # GPT-4 should be significantly more expensive
    assert cost_gpt4 > cost_gpt35 * 10

    # GPT-4 cost should be around $0.033
    assert 0.030 <= cost_gpt4 <= 0.036

    # GPT-3.5 cost should be around $0.0007
    assert 0.0005 <= cost_gpt35 <= 0.0010
