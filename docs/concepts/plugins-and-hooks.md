# Plugins and Hooks

## Extension Points

| Contract | Owns |
| --- | --- |
| `DatasetProvider` | Reading and filtering dataset items |
| `InferenceEngine` | Provider-specific prompt execution |
| `Extractor` | Parsing raw model output into structured answers |
| `Metric` | Scoring one candidate |
| `JudgeService` | Additional judge-model calls for metrics |
| `PipelineHook` | Small prompt or candidate mutations around stages |

## Recommended Boundaries

- keep data access in `DatasetProvider.scan(...)`
- keep prompt rendering in orchestration; engines should consume `trial.prompt.messages`
- keep answer parsing in extractors and `ParseSpec`
- keep metrics focused on scoring already-parsed values
- use hooks for small adjustments, not for replacing core stages

## Built-In Parsing Helpers

`PluginRegistry` auto-registers built-in extractors such as:

- `choice_letter`
- `first_number`
- `boxed_text`
- `normalized_text`
- `regex`

Use those before writing a custom extractor.

## Registering Plugins

**Individual registration** (most explicit):

```python
registry = PluginRegistry()
registry.register_inference_engine("openai", OpenAIEngine)
registry.register_metric("exact_match", ExactMatchMetric)
```

**Bulk registration with `from_dict`** (less boilerplate for common setups):

```python
registry = PluginRegistry.from_dict({
    "engines":  {"openai": OpenAIEngine, "anthropic": AnthropicEngine},
    "metrics":  {"exact_match": ExactMatchMetric, "bleu": BleuMetric},
    "extractors": {"my_parser": MyExtractor},  # supplements built-ins
})
```

Supported keys: `engines`, `metrics`, `extractors`, `judges`, `tools`, `hooks`.
Built-in extractors are always registered regardless of the mapping.

## Declaring Engine Seed Support

Use `EngineCapabilities.supports_seed` to signal whether an engine honours
the `seed` field in `InferenceParamsSpec`.  Themis uses this flag to surface
warnings when a seeded benchmark runs against an engine that ignores seeds:

```python
from themis import EngineCapabilities

registry.register_inference_engine(
    "my-engine",
    MyEngine,
    capabilities=EngineCapabilities(supports_seed=True),
)
```

When `supports_seed=False` (the default), Themis-level candidate seed derivation
still provides reproducible execution planning, but the engine itself may produce
non-deterministic outputs.
