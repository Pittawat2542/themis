# Architecture

Deep dive into Themis' internal architecture.

## System Overview

```
                    ┌─────────────────┐
                    │ themis.evaluate()│
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐
       │ Presets  │    │Generation│    │Comparison│
       │ System   │    │ Pipeline │    │  Engine  │
       └────┬─────┘    └────┬─────┘    └──────────┘
            │               │
       ┌────▼─────┐    ┌────▼─────┐
       │Benchmarks│    │Evaluation│
       │Datasets  │    │ Pipeline │
       │Prompts   │    └────┬─────┘
       └──────────┘         │
                       ┌────▼─────┐
                       │ Storage  │
                       │   V2     │
                       └──────────┘
```

## Core Components

### 1. API Layer (`themis/api.py`)

**Purpose**: Simple entry point for users

**Key Functions:**
- `evaluate()` - Main evaluation function
- Handles parameter validation
- Creates orchestrator and runs pipeline

**Design:** Facade pattern hiding complexity

```python
def evaluate(...) -> ExperimentReport:
    # 1. Parse model name
    # 2. Load preset or create custom config
    # 3. Create orchestrator
    # 4. Run pipeline
    # 5. Return report
```

### 2. Presets System (`themis/presets/`)

**Purpose**: Pre-configured benchmarks

**Components:**
- `benchmarks.py` - Benchmark registry
- `models.py` - Model name parsing

**Key Classes:**
- `BenchmarkPreset` - Benchmark configuration
- Registry pattern for extensibility

```python
@dataclass
class BenchmarkPreset:
    name: str
    prompt_template: PromptTemplate
    metrics: list[Metric]
    extractor: Extractor
    dataset_loader: Callable
    metadata_fields: tuple[str, ...]
    reference_field: str
    dataset_id_field: str
```

### 3. Generation Pipeline (`themis/generation/`)

**Purpose**: LLM inference with caching

**Components:**
- `runner.py` - Orchestrates generation
- `plan.py` - Generation plan
- `providers/` - Model providers (LiteLLM, vLLM)

**Key Features:**
- Parallel execution
- Smart caching
- Provider abstraction

**Flow:**
```
Dataset → GenerationPlan → GenerationRunner → GenerationRecords
           (with cache)      (parallel)         (stored)
```

### 4. Evaluation Pipeline (`themis/evaluation/`)

**Purpose**: Metric computation

**Components:**
- `pipeline.py` - Evaluation orchestrator
- `metrics/` - Metric implementations
  - `math/` - Math metrics
  - `nlp/` - NLP metrics
  - `code/` - Code metrics

**Key Features:**
- Modular metrics
- Cache invalidation
- Batch processing

**Flow:**
```
GenerationRecords → Extract Answers → Compute Metrics → EvaluationRecords
                     (per metric)      (parallel)        (cached)
```

### 5. Storage V2 (`themis/experiment/storage.py`)

**Purpose**: Persistent, resumable storage

**Key Features:**
- Atomic writes with `fcntl` file locking
- SQLite metadata
- Hierarchical file organization
- Smart cache invalidation

**Directory Structure:**
```
.cache/experiments/
└── {run_id}/
    ├── metadata.json
    ├── dataset/
    │   └── dataset.jsonl
    ├── generation/
    │   └── default/
    │       └── generation.jsonl
    └── evaluation/
        └── default/
            └── evaluation.jsonl
```

**Cache Keys:**
```python
# Generation cache key
cache_key = hash(
    prompt + model + temperature + max_tokens + ...
)

# Evaluation cache key
cache_key = hash(
    generation_cache_key + metrics + extractor + ...
)
```

### 6. Comparison Engine (`themis/comparison/`)

**Purpose**: Statistical comparison of runs

**Components:**
- `engine.py` - Comparison orchestrator
- `statistics.py` - Statistical tests
- `reports.py` - Result structures

**Flow:**
```
Run IDs → Load Metrics → Pairwise Tests → Win/Loss Matrix → Report
          (from storage)  (statistical)    (aggregation)    (formatted)
```

### 7. Server (`themis/server/`)

**Purpose**: Web API and dashboard

**Components:**
- `app.py` - FastAPI application
- `static/` - Web dashboard (HTML/JS)

**Endpoints:**
- REST API for querying results
- WebSocket for real-time updates
- Static dashboard for visualization

### 8. Backends (`themis/backends/`)

**Purpose**: Pluggable interfaces

**Interfaces:**
- `StorageBackend` - Custom storage
- `ExecutionBackend` - Custom execution

**Default Implementations:**
- `LocalExecutionBackend` - Multi-threaded
- `SequentialExecutionBackend` - Single-threaded

## Design Patterns

### 1. Facade Pattern

`themis.evaluate()` hides complexity:

```python
# Simple API
result = evaluate(benchmark="gsm8k", model="gpt-4")

# Internally:
# - Resolves benchmark preset
# - Creates generation plan
# - Runs orchestrator
# - Computes metrics
# - Returns report
```

### 2. Registry Pattern

Presets and providers use registries:

```python
# Register benchmark
_BENCHMARK_REGISTRY["gsm8k"] = BenchmarkPreset(...)

# Retrieve
preset = get_benchmark_preset("gsm8k")
```

### 3. Strategy Pattern

Metrics, extractors, and backends are strategies:

```python
class Metric(ABC):
    @abstractmethod
    def evaluate(self, response, reference) -> MetricScore:
        pass

# Different strategies
class ExactMatch(Metric): ...
class MathVerify(Metric): ...
class BLEU(Metric): ...
```

### 4. Builder Pattern (Legacy)

Old API used builders:

```python
# Old API (still supported in examples/)
experiment = (
    ExperimentBuilder()
    .with_dataset(...)
    .with_model(...)
    .with_metrics(...)
    .build()
)
```

**Note**: New code should use `evaluate()` directly.

### 5. Observer Pattern

Callbacks for progress monitoring:

```python
def observer(record):
    print(f"Generated: {record.id}")

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    on_result=observer,
)
```

## Data Structures

### Core Entities (`themis/core/entities.py`)

**`GenerationRecord`**
```python
@dataclass
class GenerationRecord:
    id: str
    prompt: str
    response: str
    metadata: dict
    cache_key: str
```

**`EvaluationRecord`**
```python
@dataclass
class EvaluationRecord:
    id: str
    extracted: str
    reference: str
    scores: dict[str, MetricScore]
    cache_key: str
```

**`ExperimentReport`**
```python
@dataclass
class ExperimentReport:
    run_id: str
    metrics: dict[str, float]
    num_samples: int
    cost: float
    report: str
```

### Comparison Entities (`themis/comparison/reports.py`)

**`ComparisonResult`**
```python
@dataclass
class ComparisonResult:
    metric_name: str
    run_a_id: str
    run_b_id: str
    run_a_mean: float
    run_b_mean: float
    delta: float
    winner: str
    test_result: TestResult | None
```

## Concurrency Model

### Thread Safety

Storage operations are thread-safe:

```python
# File locking for atomic writes
with _acquire_lock(file):
    write_data(file, data)
```

### Parallel Execution

Generation uses `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(generate, task) for task in tasks]
    for future in as_completed(futures):
        record = future.result()
        yield record
```

### No GIL Issues

I/O-bound operations (API calls) don't hit GIL:
- Network requests release GIL
- File I/O releases GIL
- Good scaling with threads

## Extension Points

### 1. Custom Metrics

Implement `Metric` interface:

```python
class CustomMetric(Metric):
    @property
    def name(self) -> str:
        return "CustomMetric"
    
    def evaluate(self, response, reference) -> MetricScore:
        # Your logic
        return MetricScore(value=score)
```

### 2. Custom Storage

Implement `StorageBackend`:

```python
class S3Storage(StorageBackend):
    def save_generation_record(self, run_id, record):
        # Upload to S3
        pass
    
    # ... other methods
```

### 3. Custom Execution

Implement `ExecutionBackend`:

```python
class RayExecution(ExecutionBackend):
    def map(self, func, items, **kwargs):
        # Distribute with Ray
        pass
```

### 4. Custom Extractors

Implement `Extractor` for answer extraction:

```python
class RegexExtractor(Extractor):
    def extract(self, response: str) -> str:
        # Extract with regex
        return extracted
```

## Testing Architecture

### Test Structure

Tests mirror module structure:

```
tests/
├── api/              # API tests
├── comparison/       # Comparison tests
├── backends/         # Backend tests
├── evaluation/       # Metric tests
└── generation/       # Generation tests
```

### Test Patterns

**Unit tests:**
```python
def test_exact_match():
    metric = ExactMatch()
    score = metric.evaluate("4", "4")
    assert score.value == 1.0
```

**Integration tests:**
```python
def test_full_evaluation():
    result = evaluate(
        benchmark="demo",
        model="fake-math-llm",
        limit=5,
    )
    assert result.num_samples == 5
```

## Performance Considerations

### 1. Caching

Results are cached at two levels:
- Generation cache (by prompt + model config)
- Evaluation cache (by generation + metrics)

### 2. Parallel Execution

Use workers for parallelism:
- CPU-bound: `workers = num_cpus`
- I/O-bound: `workers = 8-32`

### 3. Batching

Future improvement: Batch API requests

### 4. Memory Management

Large datasets are streamed (not loaded fully):
- JSONL format for incremental loading
- Generator-based processing

## Security

### 1. API Keys

Never commit API keys:
- Use environment variables
- Use `.env` files (not committed)
- Use secrets management in production

### 2. Arbitrary Code Execution

Be careful with:
- User-provided prompt templates
- Custom extractors
- Execution metrics (code evaluation)

### 3. File System Access

Storage uses file locking:
- Prevents concurrent write corruption
- Safe for parallel workers

## Future Architecture

### Planned Improvements

1. **Streaming API** - Real-time results
2. **Distributed Execution** - Ray/Dask integration
3. **Cloud Storage** - S3/GCS backends
4. **Database Backend** - PostgreSQL/MongoDB
5. **GPU Batching** - Efficient vLLM usage

### Extension Areas

Community can add:
- New benchmarks
- New metrics
- Backend implementations
- Provider integrations

## Further Reading

- [Storage Architecture](../STORAGE.md) - Deep dive into Storage V2
- [Extending Backends](../EXTENDING_BACKENDS.md) - Custom implementations
- [Cache Invalidation](../CACHE_INVALIDATION.md) - Caching details
- [Source Code](https://github.com/pittawat2542/themis) - Browse the code
