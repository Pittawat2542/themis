# Builtin Benchmarks

Use the builtin catalog when you want a shipped benchmark definition instead of
authoring one from scratch.

The public Python entry points are:

- `themis.catalog.list_catalog_benchmarks()`
- `themis.catalog.get_catalog_benchmark(...)`
- `themis.catalog.build_catalog_benchmark_project(...)`

Discover all shipped ids from Python:

```python
from themis.catalog import list_catalog_benchmarks

print(list_catalog_benchmarks())
```

Current builtin benchmarks:

| Benchmark | Shape | Metric | Judge | Dataset | Split |
| --- | --- | --- | --- | --- | --- |
| `aime_2025` | Short-answer math | `math_equivalence` | No | `MathArena/aime_2025` | `train` |
| `aime_2026` | Short-answer math | `math_equivalence` | No | `MathArena/aime_2026` | `train` |
| `aethercode` | Code generation with sandboxed execution | `aethercode_pass_rate` | No | `m-a-p/AetherCode` (`v1_2024`) | `test` |
| `apex_2025` | Short-answer math | `math_equivalence` | No | `MathArena/apex_2025` | `train` |
| `babe` | Binary media-bias classification | `choice_accuracy` | No | `mediabiasgroup/BABE` | `test` |
| `beyond_aime` | Short-answer math | `math_equivalence` | No | `ByteDance-Seed/BeyondAIME` | `test` |
| `encyclo_k` | Multiple choice | `choice_accuracy` | No | `m-a-p/Encyclo-K` | `test` |
| `frontierscience` | Judge-backed science reasoning | `frontierscience_score` | Yes | `openai/frontierscience` | `test` |
| `gpqa_diamond` | Multiple choice | `choice_accuracy` | No | `fingertap/GPQA-Diamond` | `test` |
| `healthbench` | Rubric-scored response | `healthbench_score` | Yes | `openai/healthbench` | `test` |
| `hle:<variant>[,<variant>...]` | Judge-backed QA | `hle_accuracy` | Yes | `cais/hle` | `test` |
| `hmmt_feb_2025` | Short-answer math | `math_equivalence` | No | `MathArena/hmmt_feb_2025` | `train` |
| `hmmt_nov_2025` | Short-answer math | `math_equivalence` | No | `MathArena/hmmt_nov_2025` | `train` |
| `humaneval[:mini|:noextreme|:vX.Y.Z|...]` | Function-level code generation with EvalPlus base tests | `humaneval_pass_rate` | No | `evalplus/HumanEvalPlus` | `test` |
| `humaneval_plus[:mini|:noextreme|:vX.Y.Z|...]` | Function-level code generation with EvalPlus base+extra tests | `humaneval_plus_pass_rate` | No | `evalplus/HumanEvalPlus` | `test` |
| `imo_answerbench` | Short-answer math | `math_equivalence` | No | `Hwilner/imo-answerbench` | `train` |
| `livecodebench` | Code generation with sandboxed execution | `livecodebench_pass_rate` | No | `livecodebench/code_generation_lite` (`release_v6`) | `test` |
| `lpfqa` | Judge-backed free-form QA | `lpfqa_score` | Yes | `m-a-p/LPFQA` | `train` |
| `mmlu_pro` | Multiple choice | `choice_accuracy` | No | `TIGER-Lab/MMLU-Pro` | `test` |
| `mmmlu[:<config>]` | Multilingual multiple choice | `choice_accuracy` | No | `openai/MMMLU` (`default` or language config) | `test` |
| `codeforces` | Code generation with sandboxed execution | `codeforces_pass_rate` | No | `open-r1/codeforces` (`verifiable-prompts`) | `test` |
| `phybench` | Short-answer physics | `math_equivalence` | No | `Eureka-Lab/PHYBench` | `train` |
| `procbench[:taskNN]` | Procedural reasoning final-answer evaluation | `procbench_final_accuracy` | No | `ifujisawa/procbench` (`task01`...`task23`) | `train` |
| `rolebench[:instruction_generalization_eng|:role_generalization_eng]` | Personality role-play response generation | `rolebench_rouge_l_f1` | No | `ZenMoore/RoleBench` | `test` |
| `simpleqa_verified` | Judge-backed short answer | `simpleqa_verified_score` | Yes | `google/simpleqa-verified` | `eval` |
| `superchem[:en|:zh]` | Multimodal chemistry multiple choice | `choice_accuracy` | No | `ZehuaZhao/SUPERChem` (`default`) | `train` |
| `supergpqa` | Multiple choice | `choice_accuracy` | No | `m-a-p/SuperGPQA` | `train` |

For sandbox setup, backend selection, runtime environment variables, and
current execution limitations for code-generation benchmarks, see
[Run Code Benchmarks](code-benchmarks.md).

Use `get_catalog_benchmark(...)` when you want to inspect or render one
definition directly:

```python
from themis.catalog import get_catalog_benchmark

definition = get_catalog_benchmark("mmlu_pro")
preview = definition.render_preview(model_id="demo-model", provider="demo")
print(preview[0]["messages"][0]["content"])
```

HLE requires explicit variants in the benchmark id. Example ids:

- `hle:text_only`
- `hle:no_tool`
- `hle:text_only,no_tool`

HumanEval supports variant tokens after `:`. Supported tokens are:

- `mini`
- `noextreme`
- one explicit dataset version such as `v0.1.10`

Examples:

- `humaneval`
- `humaneval_plus`
- `humaneval:v0.1.10`
- `humaneval_plus:mini,v0.1.10`

`mini` and `noextreme` are mutually exclusive. Duplicate or unknown tokens are
rejected.

MMMLU supports the following explicit config variants in addition to the base
`mmmlu` benchmark:

- `mmmlu:AR_XY`
- `mmmlu:BN_BD`
- `mmmlu:DE_DE`
- `mmmlu:ES_LA`
- `mmmlu:FR_FR`
- `mmmlu:HI_IN`
- `mmmlu:ID_ID`
- `mmmlu:IT_IT`
- `mmmlu:JA_JP`
- `mmmlu:KO_KR`
- `mmmlu:PT_BR`
- `mmmlu:SW_KE`
- `mmmlu:YO_NG`
- `mmmlu:ZH_CN`

Procbench supports the base aggregate `procbench` benchmark plus explicit task
variants such as:

- `procbench:task01`
- `procbench:task12`
- `procbench:task23`

RoleBench supports the aggregate English benchmark plus explicit single-variant
ids:

- `rolebench`
- `rolebench:instruction_generalization_eng`
- `rolebench:role_generalization_eng`

RoleBench scoring uses ROUGE-L F1. Install the optional dependency with
`uv add "themis-eval[text-metrics]"` if you want to run it locally.

SuperChem defaults to English. Use `superchem:zh` for the Chinese variant.

Use `build_catalog_benchmark_project(...)` when you want the full runnable
Python path for a builtin benchmark:

```python
from pathlib import Path

from themis import Orchestrator
from themis.catalog import build_catalog_benchmark_project


def load_fixture_rows(dataset_id: str, split: str, revision: str | None):
    del revision
    assert dataset_id == "TIGER-Lab/MMLU-Pro"
    assert split == "test"
    return [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "answer_index": 1,
            "category": "astronomy",
            "src": "fixture",
        }
    ]


project, benchmark, registry, dataset_provider, definition = (
    build_catalog_benchmark_project(
        benchmark_id="mmlu_pro",
        model_id="demo-model",
        provider="demo",
        storage_root=Path(".cache/themis-examples/catalog-snippet"),
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

For a runnable script version, see `examples/13_catalog_builtin_benchmark.py`.
