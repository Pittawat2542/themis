# Example Catalog

All numbered examples use the benchmark-first public surface.

| Example | Focus |
| --- | --- |
| `01_hello_world.py` | Smallest benchmark run |
| `02_project_file.py` | File-backed project policy |
| `03_custom_extractor_metric.py` | Custom parser plus metric |
| `04_compare_models.py` | Aggregation and paired comparison |
| `05_resume_run.py` | Reuse against the same storage root |
| `06_hooks_and_timeline.py` | Hooks and candidate timelines |
| `07_judge_metric.py` | Judge-backed metric |
| `08_external_stage_handoff.py` | External scoring handoff |
| `09_experiment_evolution.py` | Incremental benchmark evolution |

## Intentionally Untouched

`examples/medical_reasoning_eval` remains in the repository as a handoff and
acceptance reference. It was not rewritten to the new API, and it is not part
of the recommended public example path.
