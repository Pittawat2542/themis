# Hand Off Generation or Evaluation

The benchmark-first orchestrator still supports external stage handoffs.

## Generation Handoff

```python
bundle = orchestrator.export_generation_bundle(benchmark)
print(bundle.manifest.run_id)
print(len(bundle.items))
```

## Evaluation Handoff

```python
bundle = orchestrator.export_evaluation_bundle(benchmark)
records = build_external_trial_records(bundle)
result = orchestrator.import_evaluation_results(bundle, records)
```

The imported result is wrapped back into `BenchmarkResult` when the manifest
contains benchmark metadata.

Worked example: `examples/08_external_stage_handoff.py`
