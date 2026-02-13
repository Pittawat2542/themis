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

- `evaluate(...)` can auto-detect many hosted models and route to `litellm`.
- `ExperimentSpec.model` supports explicit routing with `"provider:model_id"`.
- For reproducibility, prefer explicit `"provider:model_id"` in specs.

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
| `vllm` | `VLLMProvider` | `vllm:meta-llama/Meta-Llama-3.1-8B-Instruct` | Use `ExperimentSession` | In-process `AsyncLLMEngine`; requires `provider_options.model` |

`Specs/session` in the table means use `ExperimentSpec(..., model="provider:model_id", ...)`.

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

### 3) In-process vLLM provider (`ExperimentSession`)

Use this when you want Themis to run vLLM directly via `AsyncLLMEngine`.

```python
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
preset = get_benchmark_preset("demo")
pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)

spec = ExperimentSpec(
    dataset=preset.load_dataset(limit=20),
    prompt=preset.prompt_template.template,
    model=f"vllm:{model_id}",
    provider_options={
        "model": model_id,               # required
        "tensor_parallel_size": 1,       # optional
        "max_parallel": 2,               # optional
        "engine_kwargs": {},             # optional
    },
    sampling={"temperature": 0.0, "top_p": 0.95, "max_tokens": 256},
    pipeline=pipeline,
    run_id="demo-vllm-inprocess",
    dataset_id_field=preset.dataset_id_field,
    reference_field=preset.reference_field,
    metadata_fields=preset.metadata_fields,
)

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=2),
    storage=StorageSpec(path=".cache/experiments", cache=True),
)
```

## Provider Options by Entry Point

- `evaluate(...)` accepts a limited set of provider kwargs (primarily endpoint/auth
  and retry/timeout fields).
- `ExperimentSpec.provider_options` passes kwargs directly to the provider
  constructor, which is the recommended path for provider-specific tuning
  (for example vLLM `engine_kwargs`).

## Quick Troubleshooting

- `RuntimeError: vLLM is not installed`:
  - install vLLM in your environment (for example `uv add vllm`)
- `TypeError: ... missing required keyword-only argument: 'model'` with `vllm`:
  - add `provider_options={"model": "<your-model-id>"}` in `ExperimentSpec`
- Hosted model auth errors:
  - confirm `api_key`/env vars and endpoint (`api_base`) are correct
