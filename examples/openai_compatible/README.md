# OpenAI-Compatible Endpoints

This example shows you how to connect Themis to real LLM endpoints using the OpenAI-compatible API. This works with:

- **LM Studio**: Local LLM server with a GUI
- **Ollama**: Lightweight local LLM runner
- **vLLM**: High-performance inference server
- **Text Generation WebUI (Oobabooga)**: Feature-rich local LLM interface
- **Any OpenAI-compatible endpoint**: Including OpenAI itself, Azure OpenAI, etc.

## What You'll Learn

1. How to configure OpenAI-compatible providers
2. How to connect to local LLM servers
3. How to use real models on real benchmarks (MATH-500)
4. How to handle API keys and authentication
5. How to configure timeouts and parallelism

## Prerequisites

### 1. Install Themis

```bash
uv pip install -e .
```

### 2. Set Up an LLM Server

Choose one of the following:

#### Option A: LM Studio (Recommended for Beginners)

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Install and open LM Studio
3. Download a model (e.g., "Qwen 2.5 7B Instruct")
4. Click "Local Server" tab
5. Click "Start Server"
6. Note the URL (usually `http://localhost:1234/v1`)

#### Option B: Ollama

```bash
# Install Ollama from ollama.ai
ollama pull qwen2.5:7b
ollama serve
# Default endpoint: http://localhost:11434/v1
```

#### Option C: vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000
# Endpoint: http://localhost:8000/v1
```

### 3. Verify Your Server is Running

```bash
# Test with curl (replace URL with your endpoint)
curl http://localhost:1234/v1/models

# Should return a list of available models
```

## Quick Start

### 1. Copy and Edit the Config

```bash
cd experiments/03_openai_compatible
cp config.sample.json my_config.json
```

Edit `my_config.json` to match your setup:

```json
{
  "run_id": "my-first-openai-run",
  "storage_dir": ".cache/my-first-openai-run",
  "resume": true,
  "n_records": 5,
  "models": [
    {
      "name": "my-local-model",
      "provider": "openai-compatible",
      "description": "My local LLM via LM Studio",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "my-local-model": "qwen2.5-7b-instruct"
        }
      }
    }
  ],
  "samplings": [
    {
      "name": "standard",
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "math500",
      "kind": "math500_hf",
      "limit": 10
    }
  ]
}
```

### 2. Run the Experiment

```bash
uv run python -m experiments.03_openai_compatible.cli run --config-path my_config.json
```

This will:
1. Download MATH-500 dataset from Hugging Face (first run only)
2. Send 10 math problems to your local LLM
3. Evaluate the responses
4. Show accuracy metrics

## Configuration Guide

### Provider Options

The `provider_options` section configures the OpenAI-compatible client:

```json
{
  "provider_options": {
    "base_url": "http://localhost:1234/v1",
    "api_key": "not-needed",
    "model_mapping": {
      "my-identifier": "actual-model-name"
    },
    "timeout": 60,
    "n_parallel": 2
  }
}
```

**Fields:**

- **`base_url`**: The OpenAI-compatible endpoint URL
  - LM Studio: `http://localhost:1234/v1`
  - Ollama: `http://localhost:11434/v1`
  - vLLM: `http://localhost:8000/v1`
  - OpenAI: `https://api.openai.com/v1`

- **`api_key`**: API key for authentication
  - Local servers: Use `"not-needed"` or any string
  - OpenAI: Use your actual API key or set `OPENAI_API_KEY` env var

- **`model_mapping`**: Maps your identifier to the server's model name
  - Key: Your identifier (used in results)
  - Value: The actual model name the server expects

- **`timeout`**: Request timeout in seconds (default: 30)
  - Increase for slower models or complex prompts

- **`n_parallel`**: Number of parallel requests (default: 4)
  - Reduce if your server can't handle concurrent requests
  - Increase for faster throughput on powerful servers

### Model Mapping Explained

The `model_mapping` lets you use friendly names in your configs:

```json
{
  "models": [
    {
      "name": "qwen-7b",  // Your identifier
      "provider": "openai-compatible",
      "provider_options": {
        "model_mapping": {
          "qwen-7b": "Qwen/Qwen2.5-7B-Instruct"  // Server's model name
        }
      }
    }
  ]
}
```

When Themis makes API calls, it uses "Qwen/Qwen2.5-7B-Instruct", but your results will reference "qwen-7b".

## Server-Specific Configurations

### LM Studio

```json
{
  "name": "lmstudio-qwen",
  "provider": "openai-compatible",
  "provider_options": {
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model_mapping": {
      "lmstudio-qwen": "qwen2.5-7b-instruct"
    },
    "timeout": 60,
    "n_parallel": 1
  }
}
```

**Tips:**
- Use `n_parallel: 1` to avoid overloading LM Studio
- Model name should match what's shown in LM Studio's "Local Server" tab
- Increase `timeout` for longer responses

### Ollama

```json
{
  "name": "ollama-qwen",
  "provider": "openai-compatible",
  "provider_options": {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model_mapping": {
      "ollama-qwen": "qwen2.5:7b"
    },
    "timeout": 45,
    "n_parallel": 2
  }
}
```

**Tips:**
- Ollama uses port 11434 by default
- Model name format: `modelname:tag` (e.g., `qwen2.5:7b`)
- List available models: `ollama list`

### vLLM

```json
{
  "name": "vllm-qwen",
  "provider": "openai-compatible",
  "provider_options": {
    "base_url": "http://localhost:8000/v1",
    "api_key": "vllm",
    "model_mapping": {
      "vllm-qwen": "Qwen/Qwen2.5-7B-Instruct"
    },
    "timeout": 30,
    "n_parallel": 8
  }
}
```

**Tips:**
- vLLM is optimized for throughput, use higher `n_parallel`
- Model name should match the HuggingFace model ID
- vLLM supports advanced sampling parameters

### Text Generation WebUI

```json
{
  "name": "textgen-model",
  "provider": "openai-compatible",
  "provider_options": {
    "base_url": "http://localhost:5000/v1",
    "api_key": "textgen",
    "model_mapping": {
      "textgen-model": "model-name"
    }
  }
}
```

**Tips:**
- Enable "OpenAI API" extension in WebUI
- Default port is 5000
- Model name is shown in WebUI's model dropdown

### OpenAI API (Real)

```json
{
  "name": "gpt-4o-mini",
  "provider": "openai-compatible",
  "provider_options": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-actual-key-here",
    "model_mapping": {
      "gpt-4o-mini": "gpt-4o-mini"
    },
    "n_parallel": 10
  }
}
```

**Tips:**
- Store API key in environment variable: `export OPENAI_API_KEY=sk-...`
- Then use: `"api_key": "${OPENAI_API_KEY}"` (requires manual substitution)
- OpenAI handles high parallelism well

## Example Configurations

### Quick Test (5 samples)

```json
{
  "run_id": "quick-test",
  "storage_dir": ".cache/quick-test",
  "resume": true,
  "n_records": 5,
  "models": [
    {
      "name": "local-model",
      "provider": "openai-compatible",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "local-model": "model-name"
        }
      }
    }
  ],
  "samplings": [
    {
      "name": "test",
      "temperature": 0.7,
      "max_tokens": 256
    }
  ],
  "datasets": [
    {
      "name": "math500",
      "kind": "math500_hf",
      "limit": 5
    }
  ]
}
```

### Temperature Comparison

Compare different sampling strategies with the same model:

```json
{
  "run_id": "temperature-sweep",
  "storage_dir": ".cache/temperature-sweep",
  "resume": true,
  "models": [
    {
      "name": "my-model",
      "provider": "openai-compatible",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "my-model": "qwen2.5-7b-instruct"
        }
      }
    }
  ],
  "samplings": [
    {
      "name": "greedy",
      "temperature": 0.0,
      "max_tokens": 512
    },
    {
      "name": "balanced",
      "temperature": 0.7,
      "max_tokens": 512
    },
    {
      "name": "creative",
      "temperature": 1.0,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "math500",
      "kind": "math500_hf",
      "limit": 50
    }
  ]
}
```

### Multi-Model Comparison

Compare different models side-by-side:

```json
{
  "run_id": "model-comparison",
  "storage_dir": ".cache/model-comparison",
  "resume": true,
  "models": [
    {
      "name": "qwen-7b",
      "provider": "openai-compatible",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "qwen-7b": "qwen2.5-7b-instruct"
        }
      }
    },
    {
      "name": "llama-8b",
      "provider": "openai-compatible",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "llama-8b": "llama-3.1-8b-instruct"
        }
      }
    }
  ],
  "samplings": [
    {
      "name": "standard",
      "temperature": 0.7,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "math500",
      "kind": "math500_hf",
      "limit": 100
    }
  ]
}
```

**Note**: Switch models in LM Studio between runs, or run multiple server instances on different ports.

## Running Experiments

```bash
# Basic run
uv run python -m experiments.03_openai_compatible.cli run --config-path my_config.json

# Dry run (preview configuration)
uv run python -m experiments.03_openai_compatible.cli run --config-path my_config.json --dry-run

# Override number of records
uv run python -m experiments.03_openai_compatible.cli run --config-path my_config.json --n-records 3

# Export results
uv run python -m experiments.03_openai_compatible.cli run \
  --config-path my_config.json \
  --csv-output results.csv \
  --html-output results.html \
  --json-output results.json

# Custom storage and run ID
uv run python -m experiments.03_openai_compatible.cli run \
  --config-path my_config.json \
  --storage-dir .cache/my-experiment \
  --run-id experiment-2024-01-15
```

## Understanding Results

After running, you'll see output like:

```
Downloading dataset: 100%|██████████| 500/500 [00:02<00:00, 200.00it/s]
Generating: 100%|██████████| 10/10 [00:45<00:00,  4.50s/it]
Evaluated 10 samples | Successful generations: 10/10 | Exact match: 0.600 (10 evaluated) | No failures
```

**Metrics explained:**
- **Total samples**: How many problems were evaluated
- **Successful generations**: How many completed without errors
- **Exact match**: Percentage of correct answers (60% in this example)
- **Failures**: Any generation or evaluation errors

### Exported Files

**CSV format** (`results.csv`):
```
sample_id,model,sampling,problem,reference,predicted,correct
math500-1,my-model,standard,"What is 2+2?",4,4,true
math500-2,my-model,standard,"What is 3*3?",9,9,true
...
```

**HTML format** (`results.html`):
- Interactive table with filtering and sorting
- Color-coded correct/incorrect answers
- View full prompts and responses

**JSON format** (`results.json`):
- Complete structured data
- Includes metadata, timestamps, and all fields
- Easy to process programmatically

## Troubleshooting

### Connection Errors

**Error**: `Connection refused` or `Failed to connect`

**Solutions**:
1. Verify the server is running: `curl http://localhost:1234/v1/models`
2. Check the port number in your config
3. Ensure no firewall is blocking the connection
4. Try `http://127.0.0.1:1234/v1` instead of `localhost`

### Model Not Found

**Error**: `Model not found` or `Invalid model`

**Solutions**:
1. Check model name in server (LM Studio: "Local Server" tab, Ollama: `ollama list`)
2. Update `model_mapping` in config to match exact name
3. Ensure model is loaded in the server

### Timeout Errors

**Error**: `Request timeout` or `Timeout waiting for response`

**Solutions**:
1. Increase `timeout` in `provider_options`
2. Reduce `max_tokens` in sampling config
3. Try a smaller/faster model
4. Reduce `n_parallel` to avoid overloading server

### Slow Performance

**Solutions**:
1. Reduce `n_parallel` if server is overwhelmed
2. Increase `n_parallel` if server can handle more
3. Use a quantized model (e.g., Q4 or Q5 versions)
4. Enable GPU acceleration in your server
5. Use `limit` to test with fewer samples first

### API Key Issues

**Error**: `Invalid API key` or `Authentication failed`

**Solutions**:
1. For local servers, try `"api_key": "not-needed"` or any string
2. For OpenAI, ensure key starts with `sk-`
3. Check for typos or extra spaces in the key
4. Use environment variables for sensitive keys

## Advanced Topics

### Using Environment Variables for API Keys

Instead of hardcoding API keys, use environment variables:

```bash
# Set the key
export OPENAI_API_KEY="sk-your-key-here"

# In your code, load it (requires custom script)
```

For production use, consider using a secrets management system.

### Custom Headers

Some servers require custom headers. Modify the provider implementation or use a proxy.

### Streaming Responses

The OpenAI-compatible provider doesn't currently support streaming, but you could extend it for real-time progress.

### Rate Limiting

When using paid APIs, be mindful of rate limits:
- Reduce `n_parallel` to avoid hitting limits
- Use `limit` to test with fewer samples
- Monitor costs with small experiments first

## Best Practices

1. **Start small**: Use `limit: 5` to verify everything works before running full benchmarks
2. **Enable resume**: Always set `resume: true` to avoid re-running completed samples
3. **Use descriptive names**: Name your models and samplings clearly for easy result interpretation
4. **Monitor server resources**: Check CPU/GPU/RAM usage during runs
5. **Export results**: Always export to CSV/HTML for easier analysis
6. **Version control configs**: Track your experiment configurations in git
7. **Document model versions**: Include model version/quantization in descriptions

## Next Steps

- **04_projects**: Organize multiple OpenAI experiments in a project structure
- **05_advanced**: Customize generation loops for advanced use cases
- Check `config.comprehensive.json` for a feature-complete example

## File Structure

```
03_openai_compatible/
├── README.md                      # This file
├── cli.py                         # CLI entry point
├── config.py                      # Configuration models
├── config.sample.json             # Quick start config
├── config.comprehensive.json      # Full-featured config
├── datasets.py                    # Dataset loaders
└── experiment.py                  # Experiment implementation
```
