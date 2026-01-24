# LiteLLM Example (Multiple Providers)

This example (`examples/litellm_example`) shows how to use Themis with various LLM providers using the LiteLLM integration.

## Key Features
- **Local LLMs**: Integration with Ollama, vLLM, or LM Studio.
- **Multiple Providers**: Switching between OpenAI, Anthropic, and Local models via config.
- **Custom Keys**: How to securely manage API keys.

## Configuration for Local LLMs

To use a local model (e.g., Qwen3 via LM Studio or vLLM):

1. **Start your server** on port 1234 (standard OpenAI compatible port).
2. **Configure Themis**:

```json
{
  "models": [
    {
      "name": "qwen/qwen3-1.7b",
      "provider": "litellm",
      "provider_options": {
        "api_base": "http://localhost:1234/v1",
        "api_key": "dummy",
        "custom_llm_provider": "openai"
      }
    }
  ]
}
```

> [!NOTE]
> The `custom_llm_provider: "openai"` option is crucial for many local servers to verify they handle the OpenAI input format correctly.

## Running the Example

```bash
# Run with the sample config (configured for local Qwen)
uv run python -m examples.litellm_example.cli run \
  --config-path examples/litellm_example/config.sample.json
```

## Supported Providers
Themis supports all providers supported by [LiteLLM](https://docs.litellm.ai/docs/providers), including:
- OpenAI
- Azure OpenAI
- Anthropic
- AWS Bedrock
- Google Vertex AI
- Hugging Face
- Ollama
- vLLM
