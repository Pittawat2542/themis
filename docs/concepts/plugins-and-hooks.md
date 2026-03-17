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
