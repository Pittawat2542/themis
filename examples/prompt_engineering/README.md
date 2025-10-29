# Prompt Engineering Example

This example demonstrates how to use Themis for systematic prompt engineering experiments. We'll test multiple prompt variations on a standard dataset using various models and evaluate with common metrics.

## Overview

This example will:

1. Define multiple prompt variations to test (zero-shot, few-shot, chain-of-thought)
2. Use a standard dataset (math/geography problems)
3. Evaluate against local or remote models
4. Compare accuracy and other metrics
5. Export results for analysis

## Running with Local LLM

The example is configured to work with a local LLM running on `http://localhost:1234/v1`. To run this example:

1. Make sure your local LLM server is running on port 1234
2. Run the experiment with the default configuration:

```bash
uv run python -m examples.prompt_engineering.cli run
```

### Example Local LLM Setup

If you're using a local model server (like LM Studio, Ollama, etc.):

```bash
# Test that your local endpoint is working:
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-vl-30b",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes. Today is Thursday" },
      { "role": "user", "content": "What day is it today?" }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
  }'
```

### Troubleshooting

If you see "Provider List" messages or all requests fail:

1. **Verify your server is running**: Test with the curl command above
2. **Check model name**: Ensure the model name in config matches what's loaded in your local server
3. **Port and URL**: Confirm your server is running on `http://localhost:1234/v1`
4. **Model compatibility**: Some local servers may need the model name to match what's actually loaded

The configuration uses these key parameters:
- `"api_base": "http://localhost:1234/v1"` - Points to your local server with proper OpenAI endpoint
- `"custom_llm_provider": "openai"` - Treats local server as OpenAI-compatible
- `"api_key": "sk-not-needed"` - Most local servers ignore this value

### Configuration

The default configuration uses:
- Model: `openai/qwen/qwen3-vl-30b` (uses `openai/` prefix for OpenAI-compatible endpoints)
- Base URL: `http://localhost:1234/v1` (includes `/v1` suffix for OpenAI-compatible endpoints)
- Temperature: 0.7 for more creative responses
- Max tokens: -1 for unlimited length responses

## Running with Different Configurations

### Using Custom Configuration File

```bash
# Run with custom configuration
uv run python -m examples.prompt_engineering.cli run --config-path config_local.json
```

### With Analysis and Export

```bash
# Run with analysis and export results
uv run python -m examples.prompt_engineering.cli run \
  --analyze \
  --csv-output results.csv \
  --html-output results.html
```

### Dry Run to Preview

```bash
# Just see what would be run without executing
uv run python -m examples.prompt_engineering.cli run --dry-run
```

## Files

- `config.py` - Configuration models for the experiment
- `experiment.py` - Core experiment implementation  
- `cli.py` - Command-line interface
- `prompts.py` - Prompt template definitions
- `datasets.py` - Dataset loading utilities
- `results_analysis.py` - Analysis utilities for results
- `config_local.json` - Example configuration for local model
- `config.sample.json` - Complete configuration template
- `USAGE.md` - Detailed usage guide