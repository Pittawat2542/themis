# Choose The Smallest Meaningful Validation

Use this file when you need to verify Themis changes without external docs.

## Default Rule

Validate at the smallest boundary that proves the change works:

- function or model unit test first
- then one focused `evaluate(...)` or `Experiment(...)` smoke path
- then CLI parity or catalog smoke only when the change is specifically about
  config loading, CLI behavior, or shipped benchmark wiring

Do not jump to a live provider or live sandbox run when deterministic builtins
or fixtures can prove the change more cheaply.

## Deterministic Wiring

Use this when the task is about authoring, parsing, scoring, or store wiring.

Prefer:

- `builtin/demo_generator`
- `builtin/json_identity`
- `builtin/text`
- `builtin/exact_match`

Smallest smoke pattern:

1. build one `Dataset` with one or two `Case` objects
2. run `evaluate(...)` or a tiny `Experiment(...)`
3. assert `status`, `run_id`, and one stored projection or metric mean

## Experiment And Config Changes

Use this when the task touches `Experiment`, config loading, or `run_id`
behavior.

Prefer:

- `Experiment.compile()` when you need to inspect identity without running
- `Experiment.from_config(...)` plus one small config fixture when the task is
  config-specific
- `sqlite` instead of `memory` when the test spans multiple commands or
  processes

If the bug is about unexpected `run_id` drift, compare compiled snapshots before
debugging runtime behavior.

## Catalog And Benchmark Changes

Use this when the task touches `themis.catalog`, shipped component ids, or
benchmark recipes.

Prefer:

- `list_component_ids(...)` or `load(...)` for registry-level checks
- fixture-backed `BenchmarkDefinition.materialize_dataset(...)` tests for
  benchmark loader behavior
- `themis.catalog.run(...)` or `themis quick-eval benchmark --name ...` only
  when you need end-to-end catalog parity

For benchmark materialization, prefer injected row loaders or fixture datasets
over live network access.

## Provider And Judge Changes

Use this when the task touches adapters, workflow metrics, or judge-model
plumbing.

Prefer:

- fake or injected clients
- deterministic demo judge components when live model behavior is irrelevant
- one focused workflow-backed metric test instead of a broad provider run

Only use a live provider when the change is explicitly about live provider
compatibility or user-specified smoke validation.

## Code-Execution Changes

Use this when the task touches reusable code-execution metrics or code
benchmarks.

Prefer:

- unit tests with a fake `SandboxExecutor`
- fixture-backed benchmark cases with small official test payloads

Use live sandboxes only when the task is explicitly about backend integration.
The default local environment variables are:

- `THEMIS_CODE_PISTON_URL`
- `THEMIS_CODE_SANDBOX_FUSION_URL`

## CLI Checks

Use CLI validation only when the task changes shell-facing behavior, config
parity, deferred execution, or quick-eval flows.

Common targeted checks:

- `themis run --config ...`
- `themis replay --config ... --stage ...`
- `themis inspect snapshot|state|evaluation --config ...`
- `themis quick-eval benchmark --name ...`

Remember that the CLI currently exposes only `generation` and `evaluation`
bundle export directly, even though Python supports reduction, parse, and score
bundle helpers too.
