# Providers and Model Connectivity

This page is the centralized reference for connecting models to every built-in
provider currently registered in Themis.

## Registered Provider Keys

Themis registers these provider keys:

- `fake`
- `litellm`
- `openai` (alias of `litellm`)
- `anthropic` (alias of `litellm`)
- `azure` (alias of `litellm`)
- `bedrock` (alias of `litellm`)
- `gemini` (alias of `litellm`)
- `cohere` (alias of `litellm`)
- `vllm`

## Model Routing Rules

- `evaluate(...)` auto-detects many hosted models and routes to `litellm`.
- For reproducibility prefer explicit `"provider:model_id"` format.

## Provider Matrix

| Provider key | Backend class | Typical model string | `evaluate(...)` support | Notes |
| --- | --- | --- | --- | --- |
| `fake` | `FakeMathModelClient` | `fake-math-llm` or `fake:fake-math-llm` | Yes | Local smoke tests, no API key needed |
| `litellm` | `LiteLLMProvider` | `gpt-4o-mini`, `claude-3-5-sonnet-20241022`, `azure/my-deployment`, `bedrock/...`, `gemini/...`, `cohere/...` | Yes | Main path for hosted and OpenAI-compatible APIs |
| `openai` | `LiteLLMProvider` | `openai:gpt-4o-mini` | Yes (explicit key) | Alias of `litellm` |
| `anthropic` | `LiteLLMProvider` | `anthropic:claude-3-5-sonnet-20241022` | Yes (explicit key) | Alias of `litellm` |
| `azure` | `LiteLLMProvider` | `azure:azure/my-deployment` | Yes (explicit key) | Alias of `litellm` |
| `bedrock` | `LiteLLMProvider` | `bedrock:bedrock/anthropic.claude-3-sonnet-20240229-v1:0` | Yes (explicit key) | Alias of `litellm` |
| `gemini` | `LiteLLMProvider` | `gemini:gemini/gemini-1.5-pro` | Yes (explicit key) | Alias of `litellm` |
| `cohere` | `LiteLLMProvider` | `cohere:cohere/command-r-plus` | Yes (explicit key) | Alias of `litellm` |
| `vllm` | `VLLMProvider` | `vllm:meta-llama/Meta-Llama-3.1-8B-Instruct` | Pass `model` in `provider_options` | In-process `AsyncLLMEngine`; requires `provider_options.model` |

## Connectivity Recipes

### 1) Hosted providers with LiteLLM (`evaluate`)

```python
from themis import evaluate

# OpenAI
evaluate("gsm8k", model="gpt-4o-mini", limit=20)

# Anthropic
evaluate("gsm8k", model="claude-3-5-sonnet-20241022", limit=20)

# Azure OpenAI (example)
evaluate(
    "gsm8k",
    model="azure/my-gpt4o-deployment",
    api_base="https://YOUR-RESOURCE.openai.azure.com",
    api_version="2024-02-01",
    api_key="YOUR_AZURE_KEY",
    limit=20,
)
```

### 2) OpenAI-compatible local servers (including vLLM server)

Use this when vLLM runs as an HTTP server.

```python
from themis import evaluate

report = evaluate(
    "demo",
    model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_base="http://127.0.0.1:8000/v1",
    api_key="dummy",
    limit=20,
)
```

### 3) In-process vLLM provider

Use this when you want Themis to run vLLM directly via `AsyncLLMEngine`.

```python
from themis import evaluate

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

report = evaluate(
    "demo",
    model=f"vllm:{model_id}",
    provider_options={
        "model": model_id,               # required
        "tensor_parallel_size": 1,       # optional
        "max_parallel": 2,               # optional
    },
    limit=20,
    run_id="demo-vllm-inprocess",
    storage_path=".cache/experiments",
)
```

## Provider Options

- `evaluate(...)` accepts provider kwargs directly (endpoint, auth, retry/timeout).
- For provider-specific constructor tuning (e.g. vLLM `engine_kwargs`) pass them
  via `provider_options`.

## Quick Troubleshooting

- `RuntimeError: vLLM is not installed`:
  - install vLLM in your environment (for example `uv add vllm`)
- `TypeError: ... missing required keyword-only argument: 'model'` with `vllm`:
  - add `provider_options={"model": "<your-model-id>"}` to `evaluate()`
- Hosted model auth errors:
  - confirm `api_key`/env vars and endpoint (`api_base`) are correct
