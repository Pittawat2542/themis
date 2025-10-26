# LiteLLM Provider Example

This example demonstrates how to use Themis with LiteLLM to connect to 100+ LLM providers including:

- **OpenAI**: GPT-4, GPT-3.5-turbo, etc.
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Azure OpenAI**: All Azure deployments
- **AWS Bedrock**: Claude, Titan, etc.
- **Google AI**: Gemini Pro, etc.
- **Local LLMs**: Ollama, LM Studio, vLLM
- **And 90+ more providers**

## What You'll Learn

1. How to configure LiteLLM for different providers
2. How to connect to cloud and local LLM servers
3. How to use real models on real benchmarks (MATH-500)
4. How to handle API keys and authentication
5. How to configure timeouts, retries, and parallelism

## Prerequisites

### 1. Install Themis

```bash
cd themis
uv sync
```

### 2. Set Up Your LLM Provider

Choose one or more of the following:

#### Option A: OpenAI (Cloud)

```bash
export OPENAI_API_KEY="sk-..."
```

#### Option B: Anthropic (Cloud)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Option C: Local with Ollama

Install and start Ollama:

```bash
# Install: https://ollama.ai/
ollama serve
ollama pull llama2
```

#### Option D: Local with LM Studio

1. Download LM Studio: https://lmstudio.ai/
2. Load a model (e.g., Llama 2 7B)
3. Start the local server (default: http://localhost:1234)

## Quick Start

### 1. Test with Fake Provider

First, verify the setup works:

```bash
uv run python -m themis.cli demo
```

### 2. Run with OpenAI

Create `config.json`:

```json
{
  "experiment_name": "openai-test",
  "storage_dir": ".cache/openai_test",
  "run_id": "run-1",
  "n_records": 5,
  "datasets": [
    {
      "type": "inline",
      "name": "math-problems",
      "rows": [
        {"unique_id": "1", "problem": "What is 2+2?", "answer": "4"},
        {"unique_id": "2", "problem": "What is 10*5?", "answer": "50"}
      ]
    }
  ],
  "models": [
    {
      "name": "gpt-3.5-turbo",
      "provider": "openai",
      "provider_options": {
        "timeout": 60,
        "max_retries": 2
      }
    }
  ],
  "samplings": [
    {
      "name": "precise",
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 200
    }
  ],
  "prompts": [
    {
      "name": "direct",
      "template": "Answer this: {problem}"
    }
  ]
}
```

Run it:

```bash
uv run python -m experiments.litellm_example.cli run --config config.json
```

### 3. Run with Multiple Providers

Compare OpenAI, Anthropic, and Google AI:

```json
{
  "models": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "provider_options": {"timeout": 90}
    },
    {
      "name": "claude-3-opus-20240229",
      "provider": "anthropic",
      "provider_options": {"timeout": 120}
    },
    {
      "name": "gemini-pro",
      "provider": "gemini",
      "provider_options": {"timeout": 60}
    }
  ]
}
```

Set API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

### 4. Run with Local LLM (Ollama)

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

## Configuration Options

### Provider Options

```json
{
  "provider_options": {
    "api_key": "sk-...",           // API key (or use env vars)
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

### Model Formats

```
# OpenAI
"gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"

# Anthropic
"claude-3-opus-20240229", "claude-3-sonnet-20240229"

# Azure
"azure/your-deployment-name"

# AWS Bedrock
"bedrock/anthropic.claude-v2"

# Google AI
"gemini-pro", "gemini-1.5-pro"

# Ollama
"ollama/llama2", "ollama/mistral"

# LM Studio (OpenAI-compatible)
Use model name with api_base: "http://localhost:1234/v1"
```

## Running Benchmarks

### MATH-500 Benchmark

The example includes MATH-500, a subset of the MATH dataset:

```bash
# Run with sample config
uv run python -m experiments.litellm_example.cli run \
  --config config.sample.json

# Run with comprehensive config (multiple models and strategies)
uv run python -m experiments.litellm_example.cli run \
  --config config.comprehensive.json
```

### Custom Dataset

Use inline datasets for quick tests:

```json
{
  "datasets": [
    {
      "type": "inline",
      "name": "my-dataset",
      "rows": [
        {
          "unique_id": "q1",
          "problem": "Your question",
          "answer": "Expected answer",
          "subject": "math",
          "level": "easy"
        }
      ]
    }
  ]
}
```

## CLI Commands

```bash
# Run experiment
uv run python -m experiments.litellm_example.cli run \
  --config config.json \
  --storage-dir .cache/my_exp \
  --run-id run-1

# Resume interrupted run
uv run python -m experiments.litellm_example.cli run \
  --config config.json \
  --run-id run-1 \
  --resume

# Limit records for testing
uv run python -m experiments.litellm_example.cli run \
  --config config.json \
  --n-records 10

# Show results
uv run python -m experiments.litellm_example.cli show \
  --storage-dir .cache/my_exp \
  --run-id run-1
```

## Example Configurations

### Local Development (LM Studio)

```json
{
  "models": [
    {
      "name": "local-model",
      "provider": "litellm",
      "provider_options": {
        "api_base": "http://localhost:1234/v1",
        "timeout": 120
      }
    }
  ]
}
```

### Production (OpenAI + Anthropic)

```json
{
  "models": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "provider_options": {
        "timeout": 90,
        "max_retries": 3,
        "n_parallel": 10
      }
    },
    {
      "name": "claude-3-opus-20240229",
      "provider": "anthropic",
      "provider_options": {
        "timeout": 120,
        "max_retries": 3,
        "n_parallel": 5
      }
    }
  ]
}
```

### Azure OpenAI

```json
{
  "models": [
    {
      "name": "azure/gpt-4-deployment",
      "provider": "azure",
      "provider_options": {
        "api_base": "https://your-endpoint.openai.azure.com/",
        "api_key": "your-azure-key"
      }
    }
  ]
}
```

## Troubleshooting

### Authentication Errors

```bash
# Make sure API keys are set
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or specify in config
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

### Connection Errors (Local)

```bash
# Verify server is running
curl http://localhost:1234/v1/models

# Check Ollama
ollama list
ollama serve
```

### Model Not Found

Check the model identifier format:
- OpenAI: `gpt-4` (not `openai/gpt-4`)
- Anthropic: `claude-3-opus-20240229`
- Azure: `azure/deployment-name`
- Ollama: `ollama/model-name`

## Next Steps

- Read [LiteLLM Provider Documentation](../../docs/LITELLM_PROVIDER.md)
- See [Migration Guide](../../docs/LITELLM_MIGRATION.md) if migrating from older configs
- Check [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for all supported providers
- Explore [example configs](.) in this directory

## Additional Resources

- **LiteLLM Documentation**: https://docs.litellm.ai/
- **Themis Documentation**: ../../README.md
- **Quick Start Guide**: ../../docs/LITELLM_QUICKSTART.md