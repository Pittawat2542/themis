# Advanced Orchestration and Customization

Themis is designed around the principle of **gradually exposing complexity**. This means that while 90% of evaluations can be run using a simple one-liner, the underlying architecture is extremely modular. When you hit the limits of the high-level API, you can drop down a level to swap out individual components, or completely manually assemble the orchestration engine.

This guide covers the four primary levels of abstraction in Themis, detailing what you can customize at each level and how to do it.

---

## Level 1: The High-Level `evaluate()` API

The highest level of abstraction is the `evaluate()` function used in conjunction with community presets. At this level, Themis auto-configures the prompts, extraction logic, metrics, and dataset loading based on a single benchmark string.

This is the fastest path to getting standard numbers on established benchmarks.

### What is controlled for you:
*   **Dataset Retrieval:** Auto-downloaded and mapped to standard properties.
*   **Prompting:** Standardized templates are applied automatically.
*   **Evaluation:** Extractor regexes and metric classes are selected for you.
*   **Orchestration:** The runner, cache manager, and reporting logic are abstracted away.

### Example

```python
from themis import evaluate

report = evaluate("math500", model="gpt-4")
print(report.evaluation_report.metrics["MathVerifyAccuracy"].mean)
```

---

## Level 2: Custom Components within `evaluate()`

When you need to test a novel dataset, craft a unique prompt strategy (like Chain-of-Thought), or use custom LLM-as-a-judge metrics, you remain within `evaluate()`, but override the preset defaults.

### What you can customize:
*   **Target Datasets:** Passing a list of Python dictionaries instead of a benchmark name.
*   **Prompt Templates:** Injecting your own `{variables}` into a string prompt.
*   **Evaluation Pipelines:** Assembling custom `Extractor` and `Metric` combinations.

### Example: Custom Dataset and Pipeline

```python
from themis import evaluate
from themis.evaluation.pipeline import EvaluationPipeline
from themis.evaluation.extractors import RegexExtractor
from my_custom_metrics import LLMJudgeMetric

custom_dataset = [
    {"id": "doc_1", "text": "Patient reports severe chest pain.", "gold": "Critical"},
    {"id": "doc_2", "text": "Patient has a mild cough.", "gold": "Routine"}
]

# Build a custom pipeline extending beyond standard exact-match
pipeline = EvaluationPipeline(
    extractor=RegexExtractor(pattern=r"Severity:\s*(Critical|Routine)"),
    metrics=[LLMJudgeMetric(judge_model="claude-3-opus-20240229")]
)

report = evaluate(
    dataset=custom_dataset,
    model="gpt-4-turbo",
    prompt="Read the notes and output 'Severity: <level>'. Notes: {text}",
    evaluation_pipeline=pipeline, # Inject pipeline
    reference_field="gold"
)
```

---

## Level 3: Pluggable Backends

When standard API rate limits, large scale generation (100k+ samples), or distributed enterprise requirements exceed the capabilities of local threading and SQLite, you can inject custom **Backends**.

You are still using `evaluate()` to orchestrate the run, but replacing the infrastructure it runs on.

### What you can customize:
*   **Execution Backend:** `themis.backends.ExecutionBackend`. Move from `ThreadPoolExecutor` to distributed architectures like Ray or Dask.
*   **Storage Backend:** `themis.backends.StorageBackend`. Move from local `.cache` folders to S3 buckets or Postgres databases.

### Example: Scaling to Distributed Execution

```python
from themis import evaluate
from themis.storage import ExperimentStorage
from my_infrastructure import RayExecutionBackend, S3StorageBackend

report = evaluate(
    "gsm8k",
    model="llama-3-70b",
    # Pass execution to a 128-core Ray cluster
    execution_backend=RayExecutionBackend(num_cpus=128),
    # Stream generations and cache to an S3 bucket 
    storage_backend=S3StorageBackend("s3://my-eval-bucket/experiments")
)
```

---

## Level 4: Manual Assembly 

The lowest level of abstraction involves bypassing `evaluate()` entirely. 

When you bypass the one-liner, you assemble the four major engine blocks yourself:
1.  **GenerationPlan**: Dictates the Cartesian product of what will be run.
2.  **GenerationRunner**: Dictates *how* it will be run (routers, retries, strategies).
3.  **EvaluationPipeline**: Dictates how responses are scored.
4.  **ExperimentOrchestrator**: The central controller that manages caching, loops, and telemetry.

### Why go this deep?
*   **Massive Sweeps:** You want to test 5 prompts against 3 models at 4 temperatures *in a single run object*, ensuring the entire hyperparameter sweep shares one Reproducibility Manifest.
*   **Agentic / Stateful Execution:** You want to evaluate complex, multi-turn agents by passing a custom `StatefulTaskExecutor` deeply into the runner loop.
*   **Highly Unorthodox Routing:** You have complex fallback or load-balancing logic that the standard `ProviderRouter` cannot handle.

### The Component Breakdown

Here is a full example of manually assembling the orchestration core.

```python
from themis.core.entities import ModelSpec, SamplingConfig
from themis.generation.plan import GenerationPlan
from themis.generation.templates import PromptTemplate
from themis.generation.router import ProviderRouter
from themis.generation.runner import GenerationRunner
from themis.providers import create_provider
from themis.evaluation.pipeline import EvaluationPipeline
from themis.evaluation.extractors import IdentityExtractor
from themis.evaluation.metrics.exact_match import ExactMatch
from themis.experiment.orchestrator import ExperimentOrchestrator
from themis.experiment.cache_manager import CacheManager
from themis.storage import ExperimentStorage

dataset = [{"question_id": 1, "q": "2+2", "expected": "4"}]

# ---------------------------------------------------------
# 1. The Generation Plan
# Dictates exactly what combinations to test.
# ---------------------------------------------------------
plan = GenerationPlan(
    templates=[
        PromptTemplate(name="zero_shot", template="{q}"),
        PromptTemplate(name="cot", template="Think step-by-step. {q}")
    ],
    models=[
        ModelSpec(identifier="gpt-4o", provider="litellm"),
        ModelSpec(identifier="claude-3-5-sonnet", provider="litellm")
    ],
    sampling_parameters=[
        SamplingConfig(temperature=0.0, max_tokens=256),
        SamplingConfig(temperature=0.7, max_tokens=1024)
    ],
    dataset_id_field="question_id",
    reference_field="expected"
)

# ---------------------------------------------------------
# 2. The Generation Runner
# Dictates how to communicate with models.
# ---------------------------------------------------------
# Map the ModelSpecs defined in the plan to actual providers
router = ProviderRouter({
    ("litellm", "gpt-4o"): create_provider("litellm"),
    ("litellm", "claude-3-5-sonnet"): create_provider("litellm")
})

runner = GenerationRunner(
    executor=router,  # Or pass a custom TaskExecutor instance directly!
    max_parallel=16,
    max_retries=3
)

# ---------------------------------------------------------
# 3. The Evaluation Pipeline
# Dictates how to score the outputs.
# ---------------------------------------------------------
pipeline = EvaluationPipeline(
    extractor=IdentityExtractor(),
    metrics=[ExactMatch()]
)

# ---------------------------------------------------------
# 4. The Orchestrator
# Binds the components together and handles state logic.
# ---------------------------------------------------------
cache = CacheManager(
    storage=ExperimentStorage(".cache/custom_experiments"),
    enable_resume=True,
    enable_cache=True
)

orchestrator = ExperimentOrchestrator(
    generation_plan=plan,
    generation_runner=runner,
    evaluation_pipeline=pipeline,
    cache_manager=cache
)

# Execute the sweep
report = orchestrator.run(
    dataset=dataset,
    run_id="model-prompt-temperature-sweep-v1",
    evaluation_batch_size=50
)

print(f"Total Evaluations Executed: {len(report.records)}")
```

By understanding these four levels, researchers can quickly iterate using `evaluate()` when exploring, and systematically step down into manual orchestration when the engineering complexity of the evaluation demands it.
