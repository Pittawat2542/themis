# Hand Off Generation or Evaluation

Use this guide when Themis should keep planning, storage, and reporting, but
generation or evaluation needs to happen in another system.

!!! note "Illustrative snippets"
    The snippets on this page show the minimum record shapes and handoff points.
    They are intentionally schematic unless an expected output block is shown.

## Choose the Handoff Shape

| Workflow | Themis entry point | External system responsibility |
| --- | --- | --- |
| Generation only inside Themis | `orchestrator.generate(experiment)` | none |
| Import generated text, then score in Themis | `import_generation_results(...)` then `evaluate(...)` | produce Themis-compatible `TrialRecord` objects with `InferenceRecord`s |
| Keep generation in Themis, score elsewhere | `export_evaluation_bundle(...)` then `import_evaluation_results(...)` | score each candidate and map outputs into `EvaluationRecord`s |
| Full external generation handoff | `export_generation_bundle(...)` then `import_generation_results(...)` | generate candidates and preserve the exported candidate IDs |

## Run Generation Only

`generate()` validates only generation-stage dependencies and materializes the
generation overlay without running transforms or metrics.

```python
generated = orchestrator.generate(experiment)
trial = generated.get_trial(generated.trial_hashes[0])
print(trial.candidates[0].inference.raw_text)
```

This is the simplest way to stop after response generation and inspect or export
the raw candidates.

## Export Missing Generation Work

When another system should do generation, export only the still-pending
generation items:

```python
generation_bundle = orchestrator.export_generation_bundle(experiment)

print(generation_bundle.manifest.run_id)
print(len(generation_bundle.items))
print(generation_bundle.items[0].candidate_id)
```

Each bundle item carries:

- the deterministic `trial_hash`
- the stable `candidate_id`
- the `TrialSpec`
- the dataset context for that item

Your external runner should preserve those IDs so the import step can attach the
responses back to the correct trials.

## Import External Generation Results

If generation happened outside Themis, convert the external payloads into
`TrialRecord` objects whose candidates contain `InferenceRecord`s:

```python
from themis.records import CandidateRecord, InferenceRecord, TrialRecord


result = orchestrator.import_generation_results(generation_bundle, trial_records)
print(result.trial_hashes)
```

Minimum imported generation fields:

- `TrialRecord.spec_hash` matching the exported `trial_hash`
- one `CandidateRecord` per exported candidate ID
- `CandidateRecord.candidate_id`
- `InferenceRecord` on each imported candidate

This is the bridge for "bring your own generation results." Once the candidates
exist in storage, you can keep using Themis for transforms, evaluation, reports,
and comparisons.

Use `examples/08_external_stage_handoff.py` for a runnable end-to-end example.

## Run Evaluation Only on Existing Candidates

If generation artifacts already exist in the storage root, `evaluate()` will
reuse them and only materialize missing transforms or evaluations:

```python
evaluated = orchestrator.evaluate(experiment)
print(evaluated.evaluation_hashes)
```

This is the simplest evaluation-only path when the candidates are already in
Themis storage, whether they were generated locally or imported.

## Export Missing Evaluation Work

When scoring or LM-judge calls should happen elsewhere, export the missing
evaluation items:

```python
evaluation_bundle = orchestrator.export_evaluation_bundle(experiment)

print(evaluation_bundle.manifest.run_id)
print(evaluation_bundle.items[0].evaluation_hash)
print(evaluation_bundle.items[0].candidate.inference.raw_text)
```

Each item includes the candidate payload that should be scored, plus the target
`evaluation_hash`. This is the handoff point for external judge pipelines,
provider batch jobs, or custom analytics services.

## Import External Evaluation Results

Map your external scores back into `EvaluationRecord` objects and import them:

```python
from themis.records import CandidateRecord, EvaluationRecord, MetricScore, TrialRecord


result = orchestrator.import_evaluation_results(evaluation_bundle, trial_records)
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
print(evaluation_result.leaderboard())
```

Minimum imported evaluation fields:

- `TrialRecord.spec_hash` matching the exported `trial_hash`
- `CandidateRecord.candidate_id` matching the exported candidate ID
- `EvaluationRecord` on each imported candidate
- at least one `MetricScore` per evaluation you want persisted

This is the bridge for "bring your own evaluation results." Your adapter script
is responsible for translating external payloads into Themis-compatible metric
rows, warnings, and optional qualitative tags in `MetricScore.details`.

## External LM Judge Workflow

A common pattern is:

1. run `generate()` locally or import external generation results
2. export evaluation items with `export_evaluation_bundle(...)`
3. call an external judge system, such as a provider batch API or custom judge
   service
4. convert each judge output into an `EvaluationRecord`
5. import those evaluation records with `import_evaluation_results(...)`
6. continue with `for_evaluation(...)`, `compare()`, `report()`, or `export_json()`

This keeps judge execution outside Themis while preserving the same downstream
inspection and reporting flow.

## Continue Into Reporting and Comparison

After external evaluation results are imported, the rest of the flow is the same
as an all-local run:

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
comparison = evaluation_result.compare(metric_id="external_exact_match")
payload = evaluation_result.export_json("external-result.json")

print(len(comparison.rows))
print(payload["overlay"])
```

Use [Compare and Export Results](compare-and-export.md) and
[Analyze Results](analyze-results.md) from that point onward.

## OpenAI Batch and Other Provider Batch APIs

!!! note
    Themis does not currently ship a built-in provider-specific adapter that
    submits OpenAI Batch jobs, polls them to completion, and automatically
    imports the results.

What Themis does support today:

- `BatchExecutionBackendSpec` for persisted async run manifests
- pending `RunHandle`s for batch backends
- exported generation/evaluation work bundles
- re-import of externally completed generation or evaluation results

What you still provide:

- the provider-specific batch submission logic
- polling, quota handling, and result download
- the adapter that converts provider outputs into Themis records

If you want a paused pipeline that resumes after a batch completes, model that
workflow as:

1. `submit(...)` or `plan(...)` to persist the manifest
2. export the relevant work bundle
3. run the external batch job
4. import the completed results
5. call `resume(run_id)` or `get_run_progress(run_id)` to confirm what work is left
