# Extending Themis Experiments

This guide covers the main extension points for adding models, datasets, tasks,
strategies, and metrics to a Themis-powered experiment.

## Models

1. **Implement/Configure a provider**
   - Either register a new `ModelProvider` class via `themis.providers.register_provider`, or
     re-use an existing provider (e.g., `fake`, `openai-compatible`).
   - For OpenAI-compatible endpoints, instantiate the provider with `base_url`
     and optional `api_key`:
     ```python
     experiment_builder.ModelBinding(
         spec=core_entities.ModelSpec(identifier="my-openai", provider="openai-compatible"),
         provider_name="openai-compatible",
         provider_options={
             "base_url": "http://localhost:1234/v1",
             "api_key": "sk-...",
         },
     )
     ```
2. **Describe model bindings**
   - In a config file, add entries to the `models` list with `name`, `provider`,
     and optional `provider_options`. The `ExperimentBuilder` turns each entry
     into a `ModelBinding` and resolves the provider by name.
3. **Plug into the builder**
   - Experiments convert configs to `ModelBinding` objects via `ExperimentDefinition`;
     no further wiring is required unless you want a custom router.

## Datasets / Benchmarks

1. **Create a dataset adapter**
   - Use helpers under `themis/datasets/` or implement a new loader that emits
     dict rows with at least an ID, prompt context, and reference answer.
2. **Reference rows in configs**
   - Add `DatasetConfig` entries (see `experiments/example/config.py`) pointing to
     the adapter (demo, HF, local). During `run_experiment`, load the rows and pass
     them to the builder/orchestrator.
3. **Resumability**
   - The orchestrator caches the entire dataset per run ID so reruns can replay
     the exact same rows even if the source isn’t available.

## Prompts / Tasks

1. **Define a `PromptTemplate`**
   - Templates live in the generation domain and interpolate row fields.
2. **Set metadata in the `GenerationPlan`**
   - Choose dataset ID/reference field names and specify `metadata_fields` for
     anything you want to propagate to providers/metrics.
3. **Agentic variations**
   - For multi-step prompts, build a custom runner (e.g., `AgenticRunner`) that
     composes new prompts/stages internally; the builder still handles the rest.

## Generation Strategies

1. **Single vs repeated sampling**
   - Use `GenerationStrategy` implementations (`SingleAttemptStrategy`,
     `RepeatedSamplingStrategy`) to control how many attempts run per task.
2. **Attach to experiments**
   - Pass a `strategy_resolver` to `ExperimentBuilder`; it receives each task and
     returns the strategy to use. The runner will expand attempts, aggregate
     results, and store attempt histories automatically.

## Evaluation Strategies and Metrics

1. **Metrics**
   - Implement subclasses of `themis.evaluation.metrics.Metric` (e.g.,
     `ResponseLength`, `MathVerifyAccuracy`). Add them to the builder’s `metrics`
     list.
2. **Evaluation strategies**
   - Provide an `evaluation_strategy_resolver` to the builder when you need
     attempt-aware scoring (e.g., average exact match across attempts).
3. **Caching**
   - Evaluation results (`EvaluationRecord`s) are cached alongside generation
     records. Rerunning with `resume=True` reuses both generation outputs and
     metric scores, so you can skip expensive re-evaluation unless the strategy
     changes.

## Storage & CLI

- Storage directories are per run ID; CLI commands expose `--storage` and
  `--run-id` flags to control caching/resume behavior. Use `--dry-run` options to
  preview plan metadata without triggering generation.
- For high-latency providers, configure parallelism: pass `runner_kwargs={"max_parallel": N}`
  to the builder, and set provider-specific limits (e.g., `openai-compatible`
  `n_parallel`) to keep request batching under control.
- Numeric benchmarks (e.g., MATH-500) can leverage math-verify by using
  `extractors.MathVerifyExtractor` and `metrics.MathVerifyAccuracy`. Install the
  `math` extra to enable these helpers.
