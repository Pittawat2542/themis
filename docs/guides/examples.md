# Example Catalog

All numbered examples use the benchmark-first public surface.

Examples that use `DatasetQuerySpec.subset(..., seed=...)` do so to make subset
selection reproducible. Omitting the seed keeps count-based sampling
deterministic and order-based rather than random.

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
| `10_agent_eval.py` | Bootstrap prompts, follow-up turns, tool declaration and selection, runtime handlers, agent traces, and persisted trace scoring |
| `11_quick_benchmark.py` | `BenchmarkSpec.simple()`, `BenchmarkSpec.preview()`, and `PluginRegistry.from_dict()` |
| `12_iter_and_estimate.py` | `run_benchmark_iter()` streaming, `estimate()` trial matrix, and `RunDiff.has_invalidated_resume_work` |
| `13_catalog_builtin_benchmark.py` | `themis.catalog.list_catalog_benchmarks()` and `build_catalog_benchmark_project(...)` with a local fixture dataset loader |
| `14_mcp_openai.py` | OpenAI Responses MCP server selection, runtime secrets, and remote tool traces |

## Intentionally Untouched

`examples/medical_reasoning_eval` remains in the repository as a handoff and
acceptance reference. It was not rewritten to the new API, and it is not part
of the recommended public example path.
