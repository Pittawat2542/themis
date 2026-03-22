# Run Code Benchmarks

Use this guide when a benchmark scores generated code by executing it against
test cases inside a sandbox.

The built-in code benchmarks today are:

- `codeforces`
- `aethercode`
- `livecodebench`

They all share the same local sandbox backends and execution runtime.

## Benchmark Matrix

| Benchmark | Dataset | Default slice | Default language | Execution modes |
| --- | --- | --- | --- | --- |
| `codeforces` | `open-r1/codeforces` | `verifiable-prompts` on `test` | Python or C++ (dataset-native) | `stdio` |
| `aethercode` | `m-a-p/AetherCode` | `v1_2024` on `test` | C++17 | `stdio` with C++ special judges |
| `livecodebench` | `livecodebench/code_generation_lite` | `release_v6` on `test` | Python 3 | `stdio` and function-call |

## What A Code Benchmark Needs

Compared with normal text benchmarks, code benchmarks add one extra runtime
requirement:

- a sandbox that can compile or run untrusted generated code

For all built-in code benchmarks, Themis currently supports two local backends:

- `piston`
- `sandbox_fusion`

Select the backend with:

```bash
export THEMIS_CODEFORCES_SANDBOX=piston
```

or:

```bash
export THEMIS_CODEFORCES_SANDBOX=sandbox_fusion
```

The environment variable names still use the historical `CODEFORCES` prefix for
backward compatibility, but they now apply to `codeforces`, `aethercode`, and
`livecodebench`.

## Backend Configuration

### Piston

Default URL:

```text
http://localhost:2000
```

Override it with:

```bash
export THEMIS_CODEFORCES_PISTON_URL=http://localhost:2000
```

Runtime selection is also configurable:

```bash
export THEMIS_CODEFORCES_PISTON_PYTHON_LANGUAGE=python
export THEMIS_CODEFORCES_PISTON_PYTHON_VERSION=3.12.0
export THEMIS_CODEFORCES_PISTON_CPP_LANGUAGE=c++
export THEMIS_CODEFORCES_PISTON_CPP_VERSION=*
```

Use Piston when you want native stdin plus argv handling without additional
wrappers.

### Sandbox Fusion

Default URL:

```text
http://localhost:8080
```

Override it with:

```bash
export THEMIS_CODEFORCES_SANDBOX_FUSION_URL=http://localhost:8080
```

Sandbox Fusion is also supported for all built-in code benchmarks. Themis sends
stdin directly, base64-encodes staged files, and wraps checker execution so
Python checkers still receive:

```text
checker.py input.txt correct_output.txt solution_output.txt
```

semantics.

## Current Benchmark Behavior

### `codeforces`

The built-in `codeforces` benchmark uses
`open-r1/codeforces` `verifiable-prompts` on the `test` split.

The dataset provider currently keeps only rows that are:

- `stdio`
- non-interactive
- backed by complete official tests

That means the current implementation intentionally skips:

- `input_mode=file`
- interactive problems
- rows where the official public tests are incomplete

This keeps the benchmark behavior aligned with the currently supported sandbox
contracts.

### `aethercode`

The built-in `aethercode` benchmark uses `m-a-p/AetherCode` subset `v1_2024`
on the `test` split.

- prompts ask for C++17 solutions
- the provider keeps only rows with published test cases
- C++ `testlib` special judges are staged automatically when a row includes a
  checker

The upstream `v1_2025` private set intentionally omits test cases, so the
shipped builtin benchmark pins the public `v1_2024` subset.

### `livecodebench`

The built-in `livecodebench` benchmark uses
`livecodebench/code_generation_lite` `release_v6` on the `test` split.

- prompts ask for Python 3 solutions
- `stdio` rows execute generated programs against decoded private tests
- function-style rows execute generated Python callables against decoded private
  tests
- the loader reads the raw `test*.jsonl` release files directly because the
  dataset repository ships a custom loading script

## Minimal CLI Flow

Preview the benchmark:

```bash
themis quick-eval benchmark \
  --benchmark codeforces \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

For a real run, use a provider-backed model and set the sandbox backend first:

```bash
export THEMIS_CODEFORCES_SANDBOX=piston

themis quick-eval benchmark \
  --benchmark livecodebench \
  --model your-model \
  --provider openai_compatible \
  --format json
```

## Minimal Python Flow

```python
from pathlib import Path

from themis import Orchestrator
from themis.catalog import build_catalog_benchmark_project


def load_fixture_rows(dataset_id: str, split: str, revision: str | None):
    del revision
    assert dataset_id == "open-r1/codeforces"
    assert split == "test"
    return [
        {
            "id": "fixture-1",
            "contest_id": "1",
            "title": "Increment",
            "description": "Read an integer and print the next integer.",
            "input_format": "One integer n.",
            "output_format": "Print n + 1.",
            "interaction_format": None,
            "time_limit": 2.0,
            "memory_limit": 256.0,
            "official_tests_complete": True,
            "official_tests": [
                {"input": "1\\n", "output": "2\\n"},
                {"input": "4\\n", "output": "5\\n"},
            ],
            "input_mode": "stdio",
            "generated_checker": None,
            "executable": True,
            "generated_tests": 0,
            "language": "python",
            "prompt": "Write a Python program that reads an integer and prints the next integer.",
            "rating": 800,
            "tags": ["implementation"],
        }
    ]


project, benchmark, registry, dataset_provider, definition = (
    build_catalog_benchmark_project(
        benchmark_id="codeforces",
        model_id="demo-model",
        provider="demo",
        storage_root=Path(".cache/themis-examples/codeforces-snippet"),
        huggingface_loader=load_fixture_rows,
    )
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_provider=dataset_provider,
)
result = orchestrator.run_benchmark(benchmark)
print(result.aggregate(group_by=["model_id", "slice_id", "metric_id"]))
print(definition.summarize_result(result))
```

## Operational Notes

- If Piston responds but `/api/v2/runtimes` is empty, benchmark execution will
  fail until a matching runtime is installed.
- `codeforces` currently supports `python`, `cpp`, and `cplusplus` labels from
  the dataset path.
- `aethercode` currently targets C++ and ships `testlib.h` for upstream special
  judges.
- `livecodebench` currently targets Python and decodes private test payloads
  from the raw benchmark files.
- Checker-based problems are supported on both backends.

For the catalog entry point and benchmark list, see
[Builtin Benchmarks](builtin-benchmarks.md).
