# Results And Operations

## Inspect Stored Results In Python

Start with the returned `BenchmarkResult`:

```python
for row in result.aggregate(group_by=["model_id", "slice_id", "metric_id"]):
    print(row)
```

For one concrete example:

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
timeline = result.view_timeline(candidate_id)

print(timeline.inference.raw_text)
```

## Compare Models And Export Artifacts

```python
comparison = result.paired_compare(
    metric_id="exact_match",
    group_by="slice_id",
)

bundle = result.persist_artifacts(storage_root=project.storage.root_dir)
```

Artifact bundle filenames are scoped by the active overlay key. Generation
views write `benchmark-aggregate-gen.json`, while evaluation views write
`benchmark-aggregate-ev-<evaluation_hash>.json`.

## Export Project And Benchmark Config Reports

```python
from themis import generate_config_report

bundle = {"project": project, "benchmark": benchmark}
markdown_report = generate_config_report(bundle, format="markdown")
```

## Quick SQLite Inspection

```bash
themis-quickcheck scores --db .cache/themis-examples/04-compare-models-benchmark-first/themis.sqlite3 --metric exact_match --slice qa
themis-quickcheck scores --db .cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3 --metric exact_match --dimension source=synthetic
themis-quickcheck failures --db .cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3 --limit 20
```

## Run Progress

```python
from themis.progress import ProgressConfig, ProgressRendererType

result = orchestrator.run_benchmark(
    benchmark,
    progress=ProgressConfig(renderer=ProgressRendererType.LOG),
)
```
