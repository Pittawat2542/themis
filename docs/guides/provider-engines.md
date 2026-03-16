# Build a Provider Engine

Use this guide when you want a real provider-backed `InferenceEngine` instead of
the fake local engines used in the examples.

!!! note "Illustrative snippets"
    The snippets on this page show the integration shape, not a reproducible
    fixed-output example. Provider responses, token counts, IDs, and latency vary
    at runtime, so this page shows success and failure patterns rather than one
    exact output transcript.

## Before You Start

Install the extra that matches your SDK:

```bash
uv add "themis-eval[providers-openai]"
```

Set credentials in environment variables before running your script:

```bash
export OPENAI_API_KEY=...
```

## Minimal Engine Skeleton

This example uses the OpenAI SDK, but the same structure applies to LiteLLM or
vLLM-backed engines.

```python
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError
from themis.records import InferenceRecord
from themis.types.enums import ErrorCode


class OpenAIChatEngine:
    def __init__(self) -> None:
        self._client = OpenAI()

    def infer(self, trial, context, runtime):
        del runtime
        try:
            response = self._client.responses.create(
                model=trial.model.model_id,
                input=[{"role": "user", "content": context["question"]}],
                max_output_tokens=trial.params.max_tokens,
            )
        except RateLimitError as exc:
            raise RetryableProviderError(
                code=ErrorCode.PROVIDER_RATE_LIMIT,
                message="Provider rate limited the request.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc
        except APIConnectionError as exc:
            raise RetryableProviderError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Provider connection failed.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc
        except AuthenticationError as exc:
            raise InferenceError(
                code=ErrorCode.PROVIDER_AUTH,
                message="Provider authentication failed.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc

        text = response.output_text
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.spec_hash}",
                raw_text=text,
                provider="openai",
                model_id=trial.model.model_id,
            )
        )
```

## Register the Engine

```python
registry.register_inference_engine("openai", OpenAIChatEngine())
```

Your `ModelSpec.provider` should use the same registry key:

```python
models = [ModelSpec(model_id="gpt-4.1-mini", provider="openai")]
```

## Error Mapping and Retries

Use stable `ErrorCode` values when mapping provider failures. That is what
project-level retry policy inspects.

Typical choices:

- `provider_timeout`
- `provider_rate_limit`
- `provider_unavailable`
- `provider_auth`

Prefer `RetryableProviderError` for transient conditions and plain
`InferenceError` for permanent failures such as authentication or malformed
requests.

Known failure pattern:

```text
InferenceError: Provider authentication failed.
```

## Practical Verification

Run your provider engine on one tiny task before scaling:

```python
estimate = orchestrator.estimate(experiment)
print(estimate.total_work_items)
result = orchestrator.run(experiment)
print(result.trial_hashes[:1])
```

Expected success output pattern:

```text
1
['<trial_hash>']
```

Then continue with:

- [Add a Minimal Plugin Set](plugins.md) for extractors and metrics
- [Scale Execution](scaling-execution.md) for concurrency and retry tuning
