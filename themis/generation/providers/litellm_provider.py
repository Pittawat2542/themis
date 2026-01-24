"""LiteLLM provider supporting 100+ LLM providers through a unified interface."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.providers import register_provider

logger = logging.getLogger(__name__)


@dataclass
class LiteLLMProvider(ModelProvider):
    """
    Universal LLM provider using LiteLLM.

    Supports 100+ providers including:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Anthropic (claude-3-opus, claude-3-sonnet, etc.)
    - Azure OpenAI (azure/<deployment-name>)
    - AWS Bedrock (bedrock/<model-id>)
    - Google AI (gemini-pro, etc.)
    - Cohere, Replicate, Hugging Face, and many more

    Configuration options:
    - api_key: Optional API key (can also use env vars like OPENAI_API_KEY)
    - api_base: Optional custom API base URL
    - timeout: Request timeout in seconds (default: 60)
    - max_retries: Number of retries for failed requests (default: 2)
    - n_parallel: Maximum number of parallel requests (default: 10)
    - drop_params: Whether to drop unsupported params (default: False)
    - custom_llm_provider: Force a specific provider (e.g., "openai", "anthropic")
    - extra_kwargs: Additional kwargs to pass to litellm.completion()
    """

    api_key: str | None = None
    api_base: str | None = None
    timeout: int = 60
    max_retries: int = 2
    n_parallel: int = 10
    drop_params: bool = False
    custom_llm_provider: str | None = None
    extra_kwargs: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._semaphore = threading.Semaphore(max(1, self.n_parallel))
        self._extra_kwargs = self.extra_kwargs or {}

        # Lazy import to avoid import errors if litellm not installed
        try:
            import litellm

            self._litellm = litellm
            # Configure litellm settings
            litellm.drop_params = self.drop_params
            if self.max_retries > 0:
                litellm.num_retries = self.max_retries
            
            logger.debug(f"LiteLLMProvider initialized:")
            logger.debug(f"  api_base: {self.api_base or 'default'}")
            logger.debug(f"  timeout: {self.timeout}s")
            logger.debug(f"  max_retries: {self.max_retries}")
            logger.debug(f"  n_parallel: {self.n_parallel}")
            
            # Warn if api_base is set but no api_key
            if self.api_base and not self.api_key:
                logger.warning(
                    "⚠️  LiteLLMProvider: api_base is set but api_key is not. "
                    "This may cause authentication errors. "
                    "Set api_key='dummy' for local servers."
                )
        except ImportError as exc:
            logger.error("❌ LiteLLM is not installed")
            raise RuntimeError(
                "LiteLLM is not installed. Install via `pip install litellm` or "
                "`uv add litellm` to use LiteLLMProvider."
            ) from exc

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        """Generate a response using LiteLLM."""

        messages = self._build_messages(task)
        completion_kwargs = self._build_completion_kwargs(task, messages)
        
        logger.debug(f"LiteLLMProvider: Calling model={completion_kwargs.get('model')}")
        if self.api_base:
            logger.debug(f"LiteLLMProvider: Using custom api_base={self.api_base}")

        try:
            with self._semaphore:
                response = self._litellm.completion(**completion_kwargs)

            # Extract the generated text
            text = response.choices[0].message.content or ""

            # Extract usage information
            usage = response.usage if hasattr(response, "usage") else None
            usage_dict = None
            metrics = {}
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

                metrics["prompt_tokens"] = prompt_tokens
                metrics["completion_tokens"] = completion_tokens
                metrics["total_tokens"] = total_tokens
                # Alias for consistency with other providers
                metrics["response_tokens"] = completion_tokens

                # Create usage dict for cost tracking
                if prompt_tokens is not None and completion_tokens is not None:
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
                    }

            # Extract model information
            model_used = getattr(response, "model", task.model.identifier)
            metrics["model_used"] = model_used

            # Convert response to dict for raw storage
            raw_data = response.model_dump() if hasattr(response, "model_dump") else {}

            return core_entities.GenerationRecord(
                task=task,
                output=core_entities.ModelOutput(text=text, raw=raw_data, usage=usage_dict),
                error=None,
                metrics=metrics,
            )

        except Exception as exc:
            # Capture detailed error information
            error_type = type(exc).__name__
            error_message = str(exc)

            # Extract additional context if available
            details: Dict[str, Any] = {
                "error_type": error_type,
                "model": task.model.identifier,
            }

            # Check for specific litellm exceptions
            if hasattr(exc, "status_code"):
                details["status_code"] = exc.status_code  # type: ignore
            if hasattr(exc, "llm_provider"):
                details["llm_provider"] = exc.llm_provider  # type: ignore
            
            # Log with helpful context
            if "AuthenticationError" in error_type or "api_key" in error_message.lower():
                logger.error(
                    f"LiteLLMProvider: ❌ Authentication error for model {task.model.identifier}"
                )
                logger.error(
                    f"  Error: {error_message[:200]}"
                )
                logger.error(
                    f"  Hint: If using a custom api_base, ensure you also pass api_key='dummy'"
                )
            elif "Connection" in error_type or "timeout" in error_message.lower():
                logger.error(
                    f"LiteLLMProvider: ❌ Connection error for model {task.model.identifier}"
                )
                logger.error(f"  Error: {error_message[:200]}")
                if self.api_base:
                    logger.error(f"  Check that the server at {self.api_base} is running")
            else:
                logger.error(
                    f"LiteLLMProvider: ❌ Generation failed for {task.model.identifier}: "
                    f"{error_type}: {error_message[:200]}"
                )

            return core_entities.GenerationRecord(
                task=task,
                output=None,
                error=core_entities.ModelError(
                    message=error_message,
                    kind=error_type,
                    details=details,
                ),
                metrics={},
            )

    def _build_messages(
        self, task: core_entities.GenerationTask
    ) -> list[dict[str, str]]:
        """Build messages array from the generation task."""
        messages = []

        # Add system message if provided in metadata
        system_prompt = task.prompt.metadata.get(
            "system_prompt"
        ) or task.prompt.spec.metadata.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add the main user prompt
        messages.append({"role": "user", "content": task.prompt.text})

        # Support for conversation history if provided
        conversation_history = task.metadata.get("conversation_history")
        if conversation_history:
            # If conversation history is provided, prepend it
            messages = conversation_history + messages

        return messages

    def _build_completion_kwargs(
        self, task: core_entities.GenerationTask, messages: list[dict[str, str]]
    ) -> Dict[str, Any]:
        """Build the kwargs dictionary for litellm.completion()."""

        kwargs: Dict[str, Any] = {
            "model": task.model.identifier,
            "messages": messages,
            "temperature": task.sampling.temperature,
            "top_p": task.sampling.top_p,
            "timeout": self.timeout,
        }

        # Add max_tokens if specified (negative values mean no limit)
        if task.sampling.max_tokens >= 0:
            kwargs["max_tokens"] = task.sampling.max_tokens

        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Add custom API base if provided
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # Add custom provider if specified
        if self.custom_llm_provider:
            kwargs["custom_llm_provider"] = self.custom_llm_provider

        # Merge any extra kwargs provided in configuration
        if self._extra_kwargs:
            kwargs.update(self._extra_kwargs)

        # Allow task-level overrides via metadata
        litellm_kwargs = task.metadata.get("litellm_kwargs", {})
        if litellm_kwargs:
            kwargs.update(litellm_kwargs)

        return kwargs


# Register the provider with multiple aliases for convenience
register_provider("litellm", LiteLLMProvider)
register_provider("openai", LiteLLMProvider)
register_provider("anthropic", LiteLLMProvider)
register_provider("azure", LiteLLMProvider)
register_provider("bedrock", LiteLLMProvider)
register_provider("gemini", LiteLLMProvider)
register_provider("cohere", LiteLLMProvider)


__all__ = ["LiteLLMProvider"]
