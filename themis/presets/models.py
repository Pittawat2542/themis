"""Model name parsing and provider detection.

This module automatically detects the appropriate provider based on
model names, eliminating the need for users to specify providers manually.
"""

from __future__ import annotations

import re
from typing import Any


def parse_model_name(model: str, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
    """Parse model name and detect provider.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus", "llama-2-70b")
        **kwargs: Additional provider-specific options
    
    Returns:
        Tuple of (provider_name, model_id, provider_options)
    
    Examples:
        >>> parse_model_name("gpt-4")
        ("litellm", "gpt-4", {})
        
        >>> parse_model_name("claude-3-opus-20240229")
        ("litellm", "claude-3-opus-20240229", {})
        
        >>> parse_model_name("local-llm", base_url="http://localhost:1234/v1")
        ("litellm", "local-llm", {"base_url": "http://localhost:1234/v1"})
    """
    model_lower = model.lower()
    
    # OpenAI models
    if any(pattern in model_lower for pattern in ["gpt-", "o1-", "text-davinci"]):
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Anthropic models
    if "claude" in model_lower:
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Google models
    if any(pattern in model_lower for pattern in ["gemini", "palm"]):
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Meta models
    if "llama" in model_lower:
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Mistral models
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Cohere models
    if "command" in model_lower and "xl" in model_lower:
        return "litellm", model, _extract_provider_options(kwargs)
    
    # AI21 models
    if "j2-" in model_lower:
        return "litellm", model, _extract_provider_options(kwargs)
    
    # Fake model for testing
    if "fake" in model_lower:
        return "fake", model, {}
    
    # Default: assume it's a litellm-compatible model
    # User can provide base_url for custom endpoints
    return "litellm", model, _extract_provider_options(kwargs)


def _extract_provider_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract provider-specific options from kwargs.
    
    Args:
        kwargs: Dictionary of options
    
    Returns:
        Dictionary of provider options
    """
    provider_options = {}
    
    # Known provider options
    option_keys = [
        "api_key",
        "base_url",
        "api_base",
        "api_version",
        "timeout",
        "max_retries",
        "n_parallel",
        "organization",
        "api_type",
        "region_name",
    ]
    
    for key in option_keys:
        if key in kwargs:
            provider_options[key] = kwargs[key]
    
    return provider_options


def get_provider_for_model(model: str) -> str:
    """Get provider name for a model (without parsing full options).
    
    Args:
        model: Model identifier
    
    Returns:
        Provider name
    
    Examples:
        >>> get_provider_for_model("gpt-4")
        "litellm"
        
        >>> get_provider_for_model("claude-3-opus")
        "litellm"
    """
    provider, _, _ = parse_model_name(model)
    return provider


# Model family detection for preset selection
def get_model_family(model: str) -> str:
    """Get the model family for capability detection.
    
    Args:
        model: Model identifier
    
    Returns:
        Model family name
    
    Examples:
        >>> get_model_family("gpt-4-turbo")
        "gpt-4"
        
        >>> get_model_family("claude-3-opus-20240229")
        "claude-3"
    """
    model_lower = model.lower()
    
    # OpenAI families
    if "gpt-4" in model_lower:
        return "gpt-4"
    if "gpt-3.5" in model_lower:
        return "gpt-3.5"
    if "o1" in model_lower:
        return "o1"
    
    # Anthropic families
    if "claude-3" in model_lower:
        if "opus" in model_lower:
            return "claude-3-opus"
        elif "sonnet" in model_lower:
            return "claude-3-sonnet"
        elif "haiku" in model_lower:
            return "claude-3-haiku"
        return "claude-3"
    if "claude-2" in model_lower:
        return "claude-2"
    
    # Google families
    if "gemini-pro" in model_lower:
        return "gemini-pro"
    if "gemini-ultra" in model_lower:
        return "gemini-ultra"
    
    # Meta families
    if "llama-2" in model_lower:
        if "70b" in model_lower:
            return "llama-2-70b"
        elif "13b" in model_lower:
            return "llama-2-13b"
        elif "7b" in model_lower:
            return "llama-2-7b"
        return "llama-2"
    if "llama-3" in model_lower:
        return "llama-3"
    
    # Mistral families
    if "mixtral" in model_lower:
        return "mixtral"
    if "mistral" in model_lower:
        return "mistral"
    
    return "unknown"


__all__ = ["parse_model_name", "get_provider_for_model", "get_model_family"]
