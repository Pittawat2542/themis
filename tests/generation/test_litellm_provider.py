"""Tests for LiteLLM provider."""

from __future__ import annotations

from unittest.mock import Mock, MagicMock
import sys
import pytest

from themis.core import entities as core_entities


def build_task(
    prompt_text: str = "Test prompt",
    model_id: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 100,
    metadata: dict | None = None,
) -> core_entities.GenerationTask:
    """Helper to build a test generation task."""
    prompt_spec = core_entities.PromptSpec(name="test", template="{text}")
    prompt = core_entities.PromptRender(
        spec=prompt_spec,
        text=prompt_text,
        context={"text": prompt_text},
        metadata=metadata or {},
    )
    model = core_entities.ModelSpec(identifier=model_id, provider="litellm")
    sampling = core_entities.SamplingConfig(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    return core_entities.GenerationTask(
        prompt=prompt, model=model, sampling=sampling, metadata=metadata or {}
    )


@pytest.fixture
def mock_litellm():
    """Mock litellm module for testing."""
    # Create a mock module
    mock_module = MagicMock()

    # Create a mock response object
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Test response"

    # Add usage information
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    mock_response.usage = mock_usage
    mock_response.model = "gpt-3.5-turbo"

    # Mock model_dump for response serialization
    mock_response.model_dump = Mock(
        return_value={
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )

    # Set up the completion function to return our mock response
    mock_module.completion = Mock(return_value=mock_response)
    mock_module.drop_params = False
    mock_module.num_retries = 2

    # Mock the module in sys.modules
    sys.modules["litellm"] = mock_module

    yield mock_module

    # Clean up
    if "litellm" in sys.modules:
        del sys.modules["litellm"]


def test_litellm_provider_basic_generation(mock_litellm):
    """Test basic generation with LiteLLM provider."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()
    task = build_task()

    record = provider.execute(task)

    assert record.output is not None
    assert record.output.text == "Test response"
    assert record.error is None
    assert record.metrics["prompt_tokens"] == 10
    assert record.metrics["completion_tokens"] == 20
    assert record.metrics["total_tokens"] == 30
    assert record.metrics["response_tokens"] == 20


def test_litellm_provider_with_custom_api_key(mock_litellm):
    """Test provider with custom API key."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(api_key="test-key-123")
    task = build_task()

    provider.execute(task)

    # Verify API key was passed to completion
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["api_key"] == "test-key-123"


def test_litellm_provider_with_custom_api_base(mock_litellm):
    """Test provider with custom API base URL."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(api_base="https://custom.api.com/v1")
    task = build_task()

    provider.execute(task)

    # Verify API base was passed to completion
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["api_base"] == "https://custom.api.com/v1"


def test_litellm_provider_with_system_prompt(mock_litellm):
    """Test provider with system prompt in metadata."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()
    task = build_task(metadata={"system_prompt": "You are a helpful assistant."})

    provider.execute(task)

    # Verify messages include system prompt
    call_kwargs = mock_litellm.completion.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"


def test_litellm_provider_sampling_parameters(mock_litellm):
    """Test that sampling parameters are correctly passed."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()
    task = build_task(temperature=0.9, top_p=0.95, max_tokens=150)

    provider.execute(task)

    # Verify sampling parameters
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["top_p"] == 0.95
    assert call_kwargs["max_tokens"] == 150


def test_litellm_provider_no_max_tokens_limit(mock_litellm):
    """Test that negative max_tokens are not passed to API."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()
    task = build_task(max_tokens=-1)

    provider.execute(task)

    # Verify max_tokens is not in kwargs when negative
    call_kwargs = mock_litellm.completion.call_args[1]
    assert "max_tokens" not in call_kwargs


def test_litellm_provider_with_extra_kwargs(mock_litellm):
    """Test provider with extra kwargs for litellm."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(
        extra_kwargs={"presence_penalty": 0.5, "frequency_penalty": 0.3}
    )
    task = build_task()

    provider.execute(task)

    # Verify extra kwargs are passed
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["presence_penalty"] == 0.5
    assert call_kwargs["frequency_penalty"] == 0.3


def test_litellm_provider_with_custom_llm_provider(mock_litellm):
    """Test provider with custom_llm_provider specified."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(custom_llm_provider="azure")
    task = build_task()

    provider.execute(task)

    # Verify custom_llm_provider is passed
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["custom_llm_provider"] == "azure"


def test_litellm_provider_error_handling(mock_litellm):
    """Test error handling when API call fails."""
    from themis.providers.litellm_provider import LiteLLMProvider

    # Configure mock to raise an exception
    mock_litellm.completion.side_effect = Exception("API Error: Rate limit exceeded")

    provider = LiteLLMProvider()
    task = build_task()

    record = provider.execute(task)

    # Verify error is captured
    assert record.output is None
    assert record.error is not None
    assert "Rate limit exceeded" in record.error.message
    assert record.error.kind == "Exception"


def test_litellm_provider_with_status_code_error(mock_litellm):
    """Test error handling with HTTP status code."""
    from themis.providers.litellm_provider import LiteLLMProvider

    # Create exception with status_code attribute
    error = Exception("API Error")
    error.status_code = 429  # type: ignore
    error.llm_provider = "openai"  # type: ignore
    mock_litellm.completion.side_effect = error

    provider = LiteLLMProvider()
    task = build_task()

    record = provider.execute(task)

    # Verify error details include status code
    assert record.error is not None
    assert record.error.details["status_code"] == 429
    assert record.error.details["llm_provider"] == "openai"


def test_litellm_provider_different_models(mock_litellm):
    """Test provider with different model identifiers."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()

    # Test various model formats
    models = [
        "gpt-4",
        "claude-3-opus-20240229",
        "azure/gpt-35-turbo",
        "bedrock/anthropic.claude-v2",
        "gemini-pro",
    ]

    for model_id in models:
        task = build_task(model_id=model_id)
        record = provider.execute(task)

        # Verify the correct model was requested
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == model_id
        assert record.output is not None


def test_litellm_provider_conversation_history(mock_litellm):
    """Test provider with conversation history."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider()

    conversation_history = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    task = build_task(
        prompt_text="What about 3+3?",
        metadata={"conversation_history": conversation_history},
    )

    provider.execute(task)

    # Verify conversation history is included
    call_kwargs = mock_litellm.completion.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 3
    assert messages[0] == conversation_history[0]
    assert messages[1] == conversation_history[1]
    assert messages[2]["content"] == "What about 3+3?"


def test_litellm_provider_task_metadata_override(mock_litellm):
    """Test that task-level metadata can override provider settings."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(timeout=30)

    task = build_task(metadata={"litellm_kwargs": {"timeout": 60, "stream": True}})

    provider.execute(task)

    # Verify task-level overrides are applied
    call_kwargs = mock_litellm.completion.call_args[1]
    assert call_kwargs["timeout"] == 60
    assert call_kwargs["stream"] is True


def test_litellm_provider_parallel_requests(mock_litellm):
    """Test that parallel request limiting works."""
    from themis.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(n_parallel=2)

    # Verify semaphore is initialized with correct value
    assert provider._semaphore._value == 2


def test_litellm_provider_empty_response(mock_litellm):
    """Test handling of empty response content."""
    from themis.providers.litellm_provider import LiteLLMProvider

    # Configure mock to return None content
    mock_litellm.completion.return_value.choices[0].message.content = None

    provider = LiteLLMProvider()
    task = build_task()

    record = provider.execute(task)

    # Verify empty string is returned
    assert record.output is not None
    assert record.output.text == ""


def test_litellm_provider_response_without_usage(mock_litellm):
    """Test handling of response without usage information."""
    from themis.providers.litellm_provider import LiteLLMProvider

    # Remove usage attribute from mock response
    del mock_litellm.completion.return_value.usage

    provider = LiteLLMProvider()
    task = build_task()

    record = provider.execute(task)

    # Verify generation succeeds without usage info
    assert record.output is not None
    assert record.output.text == "Test response"
    assert (
        "prompt_tokens" not in record.metrics or record.metrics["prompt_tokens"] is None
    )
