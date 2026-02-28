# High-Level Design

This document captures the internal module-level design of Themis: how its abstraction layers are organized, how components interact, and what design principles govern the codebase.

> For a quick overview of the public API and lifecycle, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Abstraction Layers

Themis follows a **hexagonal (ports-and-adapters)** architecture. Dependencies flow inward: public API → orchestration → domain/interfaces → infrastructure.

```
┌─────────────────────────────────────────────────────────┐
│                    Public API Layer                       │
│  themis/__init__.py, themis/api.py, themis/cli/           │
│  (User-facing entry points: evaluate(), register_*())     │
├─────────────────────────────────────────────────────────┤
│                   Orchestration Layer                      │
│  experiment/orchestrator.py, experiment/cache_manager.py   │
│  experiment/integration_manager.py                        │
│  (Coordinates generation → evaluation → reporting)        │
├─────────────────────────────────────────────────────────┤
│                     Domain Layer                          │
│  core/entities.py, core/types.py, exceptions.py           │
│  (Immutable value objects: GenerationTask, MetricScore,   │
│   Reference, ModelSpec, etc.)                             │
├─────────────────────────────────────────────────────────┤
│         Interfaces / Ports Layer (Hexagonal Core)         │
│  interfaces/__init__.py                                   │
│  (ABCs/Protocols: StatelessTaskExecutor, Metric,          │
│   DatasetAdapter, Extractor, StatefulTaskExecutor)        │
├──────────────┬──────────────┬───────────────────────────┤
│  Generation   │  Evaluation   │  Comparison               │
│  Subsystem    │  Subsystem    │  Subsystem                │
│  gen/runner   │  eval/pipeline│  comparison/engine.py     │
│  gen/plan     │  eval/metrics │  comparison/reports.py    │
│  gen/batching │  eval/extract │  eval/statistics/*        │
│  gen/strats   │  eval/reports │                           │
│               │  eval/cond.   │                           │
├──────────────┴──────────────┴───────────────────────────┤
│                 Infrastructure Layer                      │
│  storage/core.py, storage/database.py, storage/locking    │
│  backends/execution.py, backends/storage.py               │
│  config/runtime.py, config/schema.py                      │
│  providers/registry.py, datasets/registry.py              │
│  integrations/ (langfuse, wandb, huggingface)             │
├─────────────────────────────────────────────────────────┤
│                    Presentation Layer                      │
│  experiment/export.py (CSV, HTML, JSON, LaTeX)            │
│  experiment/visualization.py, server/app.py               │
│  presets/benchmarks.py (benchmark configs)                 │
│  utils/ (dashboard, progress, logging, tracing)           │
└─────────────────────────────────────────────────────────┘
```

## Component Interaction

The following diagram shows the primary data flow during a `themis.evaluate()` call:

```
 User
  │
  ▼
 themis.evaluate()  ──── resolves benchmarks (presets/)
  │                 ──── resolves datasets (datasets/registry)
  │                 ──── resolves metrics (evaluation/metric_resolver)
  │                 ──── creates provider (providers/registry)
  │
  ▼
 ExperimentOrchestrator
  │
  ├──→ GenerationPlan.expand(dataset)  →  Iterator[GenerationTask]
  │
  ├──→ CacheManager  ←──→  ExperimentStorage (storage/core.py)
  │       │                     ├── FileSystem (storage/filesystem.py)
  │       │                     ├── LockManager (storage/locking.py)
  │       │                     └── DatabaseIndex (storage/database.py)
  │
  ├──→ GenerationRunner.run(tasks)  ──→ StatelessTaskExecutor.execute()
  │       │                                 ├── LiteLLMExecutor
  │       │                                 ├── VLLMExecutor
  │       │                                 └── FakeMathModelClient
  │       └── ThreadPoolExecutor (parallel, retries, backoff)
  │
  ├──→ EvaluationPipeline.evaluate(records)
  │       ├── Extractor.extract(raw_output)
  │       └── Metric.compute(prediction, references)
  │
  ├──→ IntegrationManager
  │       ├── WandBTracker
  │       ├── LangfuseTracker
  │       └── HuggingFaceUploader
  │
  └──→ ExperimentReport  →  export_* (CSV/HTML/JSON)
```

## Interface Design Rationale

The `interfaces/` module mixes **ABC** and **Protocol** styles intentionally:

- **ABCs** (`StatelessTaskExecutor`, `StatefulTaskExecutor`, `Metric`): Used when the contract includes class-level attributes (e.g., `Metric.name`, `Metric.requires_reference`) or when explicit inheritance is desirable for documentation and IDE support.
- **Protocols** (`DatasetAdapter`, `Extractor`): Used for simpler single-method contracts where duck typing ("just implement `extract()`") is more Pythonic and reduces coupling.

This means implementing `Metric` requires explicit inheritance (`class MyMetric(Metric)`), while implementing `Extractor` works via duck typing alone.

## Registry Pattern

All four extension points follow the same module-level singleton pattern:

```python
# Internal registry class
class _FooRegistry:
    def register(self, name, factory): ...
    def create(self, name, **options): ...
    def list(self): ...

# Module-level singleton
_REGISTRY = _FooRegistry()

# Public functions delegating to singleton
def register_foo(name, factory): _REGISTRY.register(name, factory)
def create_foo(name, **options): return _REGISTRY.create(name, **options)
def list_foos(): return _REGISTRY.list()
```

Applied consistently to: `providers/registry.py`, `datasets/registry.py`, `evaluation/metric_resolver.py`, and `presets/benchmarks.py`.

## Exception Hierarchy

All domain exceptions inherit from both `ThemisError` and a stdlib counterpart, making error handling backward-compatible:

```
ThemisError (base)
├── ConfigurationError (ThemisError, ValueError)
├── ProviderError (ThemisError, KeyError)
├── DatasetError (ThemisError, ValueError)
├── MetricError (ThemisError, ValueError)
├── EvaluationError (ThemisError, RuntimeError)
├── StorageError (ThemisError, RuntimeError)
└── DependencyError (ThemisError, ImportError)
```

## Domain Entities

All domain value objects in `core/entities.py` use `@dataclass(frozen=True)` for immutability (except `GenerationRecord`, `EvaluationRecord`, and `ExperimentReport`, which accumulate mutable state during pipeline execution).

Key entities and their relationships:

```
ModelSpec + PromptRender + SamplingConfig → GenerationTask
GenerationTask → (via executor) → GenerationRecord (output + error)
GenerationRecord → (via pipeline) → EvaluationRecord (scores + failures)
EvaluationRecord* → EvaluationReport → ExperimentReport
```

## Current Design Trade-offs

- **`api.py` as God Function**: The `evaluate()` function handles all configuration resolution and wiring in a single 320-line function. This trades maintainability for a simple call signature.
- **Module-level side effects for registration**: Provider modules register themselves at import time (e.g., `register_provider("fake", FakeMathModelClient)` at module top-level), with lazy import guards ensuring correct ordering.
- **Dual comparison subsystems**: `comparison/engine.py` (statistical analysis) and `experiment/comparison.py` (data structure + export) exist as separate modules with related but non-overlapping responsibilities.
