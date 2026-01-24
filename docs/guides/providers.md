# LLM Providers (LiteLLM Integration)

Themis uses [LiteLLM](https://github.com/BerriAI/litellm) to support 100+ LLM providers, including OpenAI, Anthropic, Azure, AWS Bedrock, and local models.

## Standard "Local LLM" Configuration

To use a local model (e.g., via LM Studio, vLLM, or Ollama) with an OpenAI-compatible endpoint:

```json
{
  "models": [
    {
      "name": "qwen/qwen3-1.7b",  // Your local model name
      "provider": "litellm",
      "provider_options": {
        "api_base": "http://localhost:1234/v1",  // Your local server URL
        "api_key": "dummy",                      // Required non-empty string
        "custom_llm_provider": "openai"          // Forces OpenAI protocol
      }
    }
  ]
}
```

> [!TIP]
> Always set `custom_llm_provider="openai"` for generic local endpoints to ensure maximum compatibility.

## Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Azure OpenAI**: All Azure deployments
- **AWS Bedrock**: Claude, Titan, etc.
- **Google AI**: Gemini Pro, Gemini 1.5 Pro
- **Cohere**: Command R, Command R+
- **Local Models**: Ollama, LM Studio, vLLM
- **And 90+ more**: See [LiteLLM docs](https://docs.litellm.ai/docs/providers)

## Quick Start

### 1. Set API Key

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google AI
export GEMINI_API_KEY="..."
```

### 2. Configure Model

```json
{
  "models": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "provider_options": {
        "timeout": 60,
        "max_retries": 2,
        "n_parallel": 10
      }
    }
  ]
}
```

### 3. Run Experiment

```bash
uv run python -m examples.litellm_example.cli run --config config.json
```

## Configuration Options

```json
{
  "provider_options": {
    "api_key": "sk-...",           // Optional if using env vars
    "api_base": "...",             // Custom API endpoint
    "timeout": 60,                 // Request timeout (seconds)
    "max_retries": 2,              // Number of retries
    "n_parallel": 10,              // Max parallel requests
    "drop_params": false,          // Drop unsupported params
    "custom_llm_provider": "...",  // Force specific provider
    "extra_kwargs": {}             // Additional parameters
  }
}
```

## Provider Examples

### OpenAI

```json
{
  "name": "gpt-4",
  "provider": "openai",
  "provider_options": {
    "timeout": 60
  }
}
```

Environment: `OPENAI_API_KEY`

### Anthropic

```json
{
  "name": "claude-3-opus-20240229",
  "provider": "anthropic",
  "provider_options": {
    "timeout": 120
  }
}
```

Environment: `ANTHROPIC_API_KEY`

### Azure OpenAI

```json
{
  "name": "azure/my-deployment",
  "provider": "azure",
  "provider_options": {
    "api_base": "https://my-endpoint.openai.azure.com/",
    "api_key": "..."
  }
}
```

Or use environment variables:
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://..."
export AZURE_API_VERSION="2024-02-15-preview"
```

### AWS Bedrock

```json
{
  "name": "bedrock/anthropic.claude-v2",
  "provider": "bedrock"
}
```

Environment:
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"
```

### Google AI

```json
{
  "name": "gemini-pro",
  "provider": "gemini",
  "provider_options": {
    "timeout": 60
  }
}
```

Environment: `GEMINI_API_KEY`

### Local Ollama

```json
{
  "name": "ollama/llama2",
  "provider": "litellm",
  "provider_options": {
    "api_base": "http://localhost:11434",
    "custom_llm_provider": "ollama"
  }
}
```

Start Ollama:
```bash
ollama serve
ollama pull llama2
```

### Local LM Studio

```json
{
  "name": "my-local-model",
  "provider": "litellm",
  "provider_options": {
    "api_base": "http://localhost:1234/v1",
    "timeout": 120
  }
}
```

Example for user's local Qwen3 model:
```json
{
  "name": "qwen/qwen3-1.7b",
  "provider": "openai",
  "provider_options": {
    "api_base": "http://localhost:1234/v1",
    "api_key": "dummy",
    "custom_llm_provider": "openai"
  }
}
```

## Model Identifier Formats

```
OpenAI:       "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"
Anthropic:    "claude-3-opus-20240229", "claude-3-sonnet-20240229"
Azure:        "azure/your-deployment-name"
Bedrock:      "bedrock/anthropic.claude-v2"
Google:       "gemini-pro", "gemini-1.5-pro"
Ollama:       "ollama/llama2", "ollama/mistral"
LM Studio:    Use any name with api_base set
```

## Features

### Automatic Retries

```json
{
  "provider_options": {
    "max_retries": 3
  }
}
```

### System Prompts

```json
{
  "prompts": [
    {
      "name": "expert",
      "template": "Solve: {problem}",
      "metadata": {
        "system_prompt": "You are an expert mathematician."
      }
    }
  ]
}
```

### Extra Parameters

```json
{
  "provider_options": {
    "extra_kwargs": {
      "presence_penalty": 0.5,
      "frequency_penalty": 0.3,
      "response_format": {"type": "json_object"}
    }
  }
}
```

## Metrics

The provider tracks:

- `prompt_tokens`: Tokens in prompt
- `completion_tokens`: Tokens in completion
- `total_tokens`: Total tokens used
- `response_tokens`: Alias for completion_tokens
- `model_used`: Actual model that processed the request

Access via:

```python
record = provider.generate(task)
print(f"Tokens: {record.metrics['total_tokens']}")
print(f"Model: {record.metrics['model_used']}")
```

## Error Handling

Errors are captured in `GenerationRecord.error`:

```python
if record.error:
    print(f"Error: {record.error.message}")
    print(f"Type: {record.error.kind}")
    print(f"Status: {record.error.details.get('status_code')}")
    print(f"Provider: {record.error.details.get('llm_provider')}")
```

## Troubleshooting

### Authentication Error

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or add to config
{"provider_options": {"api_key": "sk-..."}}
```

### Rate Limiting

```json
{
  "provider_options": {
    "n_parallel": 2,
    "max_retries": 5,
    "timeout": 120
  }
}
```

### Unsupported Parameters

```json
{
  "provider_options": {
    "drop_params": true
  }
}
```

### Model Not Found

Check the model identifier format for your provider:
- OpenAI: `"gpt-4"` (not `"openai/gpt-4"`)
- Azure: `"azure/deployment-name"`
- Bedrock: `"bedrock/model-id"`
- Ollama: `"ollama/model-name"`

## Provider Aliases

The following provider names all use LiteLLM:

- `litellm` - Generic alias
- `openai` - For OpenAI models
- `anthropic` - For Anthropic Claude
- `azure` - For Azure OpenAI
- `bedrock` - For AWS Bedrock
- `gemini` - For Google AI
- `cohere` - For Cohere models

## Examples

### Compare Multiple Providers

```json
{
  "models": [
    {"name": "gpt-4", "provider": "openai"},
    {"name": "claude-3-opus-20240229", "provider": "anthropic"},
    {"name": "gemini-pro", "provider": "gemini"}
  ],
  "samplings": [
    {"name": "precise", "temperature": 0.0, "top_p": 1.0, "max_tokens": 200}
  ]
}
```

Set all API keys:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

### Local Development

```json
{
  "models": [
    {
      "name": "ollama/llama2",
      "provider": "litellm",
      "provider_options": {
        "api_base": "http://localhost:11434",
        "custom_llm_provider": "ollama"
      }
    }
  ]
}
```

## See Also

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported Providers](https://docs.litellm.ai/docs/providers)
- [Example Configs](../examples/litellm_example/)
- [Themis Main Documentation](index.md)