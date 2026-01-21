# Advanced Customization

This example shows you how to customize and extend Themis's default behaviors. You'll learn how to:

1. Override the generation loop with custom runners
2. Create custom evaluation pipelines
3. Implement custom metrics and extractors
4. Add instrumentation and tracking
5. Build agentic workflows with multi-step generation

This is the most advanced example and assumes you're comfortable with the basics from examples 01-04.

## What You'll Learn

- Custom generation runners that control execution order
- Subject-aware evaluation pipelines with hierarchical metrics
- Instrumented provider routers for debugging
- Agentic workflows with planning and execution phases
- Advanced prompt engineering patterns

## Prerequisites

```bash
uv pip install -e .
```

## Overview of Customization Points

Themis is designed to be extended at multiple levels:

```
ExperimentOrchestrator
├── GenerationRunner (customizable)
│   ├── ProviderRouter (customizable)
│   └── Generation loop (customizable)
├── EvaluationPipeline (customizable)
│   ├── Extractors (customizable)
│   └── Metrics (customizable)
└── Storage (customizable with StorageConfig)
```

## Example 0: Optimized Storage Configuration

### Problem

Large experiments can generate substantial storage overhead. By default, Themis saves:
- Full API responses (~3-5KB each)
- Duplicate prompt templates in every task
- Uncompressed JSONL files

For 10,000 samples, this can result in ~120MB of storage.

### Solution: StorageConfig

Configure storage optimization for your needs:

```python
from themis.experiment.storage import ExperimentStorage, StorageConfig
from themis.experiment.export import export_summary_json, export_report_bundle

# Production: Maximum optimization (recommended)
storage_config = StorageConfig(
    save_raw_responses=False,    # Skip full API responses (saves ~5MB per 1.5K samples)
    compression="gzip",           # Enable compression (50-60% reduction)
    deduplicate_templates=True,   # Store templates once (saves ~627KB per 1.5K samples)
    save_dataset=False,           # Don't duplicate dataset if loading from file
)

# Development: Balanced approach
storage_config = StorageConfig(
    save_raw_responses=False,
    compression="gzip",
    deduplicate_templates=True,
)

# Debug: Keep everything for inspection
storage_config = StorageConfig(
    save_raw_responses=True,
    compression="none",
    deduplicate_templates=False,
)

# Use in experiment builder
storage = ExperimentStorage(".cache/experiments", config=storage_config)
```

### Quick Summary Export

Export lightweight summaries for fast result viewing:

```python
from themis.experiment.export import export_summary_json, export_report_bundle

# After running experiment
report = run_experiment(config)

# Export summary (1KB) for quick viewing
export_summary_json(
    report,
    f"{config.storage_dir}/{config.run_id}/summary.json",
    run_id=config.run_id
)

# Or export everything including summary
export_report_bundle(
    report,
    json_path=f"{config.storage_dir}/{config.run_id}/report.json",
    summary_path=f"{config.storage_dir}/{config.run_id}/summary.json",
    csv_path=f"{config.storage_dir}/{config.run_id}/results.csv",
    run_id=config.run_id
)

# View summaries via CLI
# uv run python -m themis.cli results-summary --run-id run-123
# uv run python -m themis.cli results-list
```

### Storage Savings

**Before optimizations (1,500 samples):**
```
outputs/run-id/
├── dataset.jsonl      604KB
├── tasks.jsonl        2.5MB  (templates duplicated)
├── records.jsonl      13MB   (raw responses included)
├── evaluation.jsonl   772KB
└── report.json        1.6MB
Total: 18.5MB
```

**After optimizations (1,500 samples):**
```
outputs/run-id/
├── templates.jsonl.gz     0.4KB  (deduplicated!)
├── tasks.jsonl.gz         500KB  (compressed, references templates)
├── records.jsonl.gz       2MB    (compressed, no raw responses)
├── evaluation.jsonl.gz    200KB  (compressed)
├── summary.json           1KB    (quick access!)
└── report.json            1.6MB  (optional)
Total: ~4.3MB (77% reduction!)
```

**For 10,000 samples:**
- Before: ~120MB
- After: ~20-30MB
- Savings: ~90-100MB (75% reduction)

## Example 1: Custom Generation Runner

### Problem

The default generation runner processes tasks in arbitrary order. For some use cases, you might want to:

- Batch tasks by subject to reduce context switching
- Prioritize certain samples
- Implement custom retry logic
- Add preprocessing steps

### Solution: PrioritizedGenerationRunner

This custom runner sorts tasks by subject and model before processing:

```python
from typing import Iterable, Iterator
from themis.core import entities as core_entities
from themis.generation.runner import GenerationRunner

class PrioritizedGenerationRunner(GenerationRunner):
    """Batches generation tasks by subject and model to reduce switching costs."""

    def __init__(
        self,
        *,
        provider,
        priority_field: str = "subject",
        chunk_size: int = 4,
        **kwargs,
    ):
        super().__init__(provider=provider, **kwargs)
        self._priority_field = priority_field
        self._chunk_size = max(1, chunk_size)

    def run(
        self, tasks: Iterable[core_entities.GenerationTask]
    ) -> Iterator[core_entities.GenerationRecord]:
        # Convert to list and sort by priority
        task_list = list(tasks)
        task_list.sort(key=self._priority_key)
        
        # Process in chunks
        for chunk in self._chunk(task_list, self._chunk_size):
            for record in super().run(chunk):
                yield record

    def _priority_key(self, task: core_entities.GenerationTask):
        subject = str(task.metadata.get(self._priority_field, ""))
        model = task.model.identifier
        dataset_id = str(task.metadata.get("dataset_id", ""))
        return (subject, model, dataset_id)
    
    def _chunk(self, sequence, size):
        for index in range(0, len(sequence), size):
            yield sequence[index : index + size]
```

### Usage

```python
from themis.experiment.builder import ExperimentBuilder
from .generation import PrioritizedGenerationRunner

# Create builder with custom runner
builder = ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[metrics.ExactMatch()],
    runner_factory=lambda provider: PrioritizedGenerationRunner(
        provider=provider,
        priority_field="subject",
        chunk_size=4,
    ),
)

built = builder.build(definition, storage_dir=".cache/prioritized")
report = built.orchestrator.run(dataset=dataset, run_id="prioritized-run")
```

**Benefits:**
- Reduces model/context switching overhead
- Better cache locality for similar problems
- More predictable execution order
- Easier debugging with grouped outputs

## Example 1.5: Multi-Value References and Custom Selectors

### Problem

Some tasks require multiple reference values for evaluation. For example:
- Countdown task: needs target number AND allowed numbers
- Math problem: needs answer AND valid solution steps
- Code generation: needs output AND test cases

The simple single-value Reference doesn't support this directly.

### Solution: Dict-Valued References

Use dict values in References to store multiple related values:

```python
from themis.core import entities as core_entities

# Create task with multi-value reference
task = core_entities.GenerationTask(
    prompt=core_entities.PromptRender(
        spec=core_entities.PromptSpec(
            name="countdown",
            template="Using {numbers_str}, make {target}"
        ),
        text="Using 25, 50, 75, 100, make 122",
        context={"numbers_str": "25, 50, 75, 100", "target": 122}
    ),
    model=model_spec,
    sampling=sampling_config,
    reference=core_entities.Reference(
        kind="countdown",
        value={
            "target": 122,
            "numbers": [25, 50, 75, 100]
        }
    ),
    metadata={"numbers": [25, 50, 75, 100]}
)
```

### Custom Reference Selector

Extract multi-value references from task metadata:

```python
from themis.evaluation import EvaluationPipeline

def countdown_reference_selector(record):
    """Extract both target and numbers from task."""
    return {
        "target": record.task.reference.value,
        "numbers": record.task.metadata.get("numbers", [])
    }

# Create pipeline with custom selector
pipeline = EvaluationPipeline(
    extractor=my_extractor,
    metrics=[countdown_metric],
    reference_selector=countdown_reference_selector
)

# Note: You'll see a warning about using custom selector with DefaultEvaluationStrategy
# This is normal - the selector will work correctly and take precedence
```

### Using in Metrics

Access multi-value references in your metric:

```python
from themis.interfaces import Metric
from themis.core import entities

class CountdownMetric(Metric):
    name = "countdown_accuracy"
    
    def compute(self, *, prediction, references, metadata=None):
        # references is always a list (normalized by pipeline)
        ref = references[0]
        
        # Check if it's a dict (multi-value reference)
        if isinstance(ref, dict):
            target = ref["target"]
            numbers = ref["numbers"]
        else:
            # Fallback for simple reference
            target = ref
            numbers = metadata.get("numbers", [])
        
        # Validate prediction uses only allowed numbers to reach target
        is_valid = self.validate_solution(prediction, numbers, target)
        
        return entities.MetricScore(
            metric_name=self.name,
            value=1.0 if is_valid else 0.0,
            details={
                "target": target,
                "allowed_numbers": numbers,
                "prediction": prediction
            }
        )
    
    def validate_solution(self, prediction, numbers, target):
        # Your validation logic here
        # Check if prediction uses only numbers from 'numbers' to reach 'target'
        pass
```

### Pattern: Multiple Valid Answers

Return a list from reference selector for multiple valid answers:

```python
def multiple_answers_selector(record):
    """Allow multiple valid answers."""
    primary = record.task.reference.value
    alternatives = record.task.metadata.get("alternative_answers", [])
    # Return as list - metric will check against all
    return [primary] + alternatives

# In metric
def compute(self, *, prediction, references, metadata=None):
    # references is list of valid answers
    is_correct = any(
        prediction.strip().lower() == str(ref).strip().lower()
        for ref in references
    )
    return entities.MetricScore(
        metric_name=self.name,
        value=1.0 if is_correct else 0.0
    )
```

### Best Practices

1. **Use dict for related values:**
   ```python
   # ✅ Good: Related values together
   Reference(kind="task", value={"answer": 42, "steps": [...]})
   
   # ❌ Avoid: Scattered across metadata
   Reference(kind="answer", value=42)
   metadata={"steps": [...]}
   ```

2. **Handle both formats gracefully:**
   ```python
   ref = references[0]
   if isinstance(ref, dict):
       # Multi-value reference
       answer = ref["answer"]
   else:
       # Simple reference
       answer = ref
   ```

3. **Document expected reference format:**
   ```python
   class MyMetric(Metric):
       """Custom metric.
       
       Expected reference format:
       - Simple: string or int
       - Complex: dict with keys: "answer", "steps", "constraints"
       """
   ```

## Example 2: Custom Evaluation Pipeline

### Problem

The default evaluation pipeline computes overall metrics. You might want:

- Per-subject or per-category breakdowns
- Hierarchical metric aggregation
- Custom metric computation logic
- Additional metadata in reports

### Solution: SubjectAwareEvaluationPipeline

This custom pipeline computes both overall and per-subject metrics:

```python
from collections import defaultdict
from statistics import mean
from typing import Sequence
from themis.core import entities as core_entities
from themis.evaluation.pipeline import EvaluationPipeline, MetricAggregate

class SubjectAwareEvaluationPipeline(EvaluationPipeline):
    """Extends the base evaluation pipeline with per-subject aggregates."""

    def __init__(self, *args, subject_field: str = "subject", **kwargs):
        super().__init__(*args, **kwargs)
        self._subject_field = subject_field
        self.subject_breakdown = {}

    def evaluate(self, records: Sequence[core_entities.GenerationRecord]):
        # First, run standard evaluation
        report = super().evaluate(records)
        
        if not records:
            self.subject_breakdown = {}
            return report

        # Build lookup of scores by sample ID
        subject_scores = defaultdict(list)
        score_lookup = {}
        
        exact_metric = report.metrics.get("ExactMatch")
        if exact_metric:
            for score in exact_metric.per_sample:
                sample_id = score.metadata.get("sample_id")
                if sample_id is not None:
                    score_lookup[str(sample_id)] = score.value

        # Group scores by subject
        for record in records:
            sample_id = str(record.task.metadata.get("dataset_id"))
            subject = str(record.task.metadata.get(self._subject_field, "unknown"))
            value = score_lookup.get(sample_id)
            if value is not None:
                subject_scores[subject].append(value)

        # Compute per-subject metrics
        subject_metrics = []
        for subject, values in subject_scores.items():
            avg = mean(values)
            subject_metrics.append(
                core_entities.MetricScore(
                    metric_name="SubjectExactMatch",
                    value=avg,
                    metadata={"subject": subject, "count": len(values)},
                    details={},
                )
            )
        
        # Add to report
        if subject_metrics:
            report.metrics["SubjectExactMatch"] = MetricAggregate(
                name="SubjectExactMatch",
                count=len(subject_metrics),
                mean=mean(score.value for score in subject_metrics),
                per_sample=subject_metrics,
            )
        
        # Store breakdown for easy access
        self.subject_breakdown = {
            score.metadata["subject"]: score.value 
            for score in subject_metrics
        }
        
        return report
```

### Usage

```python
from themis.experiment.builder import ExperimentBuilder
from .pipeline import SubjectAwareEvaluationPipeline

# Create builder with custom pipeline
builder = ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[metrics.ExactMatch()],
    pipeline_factory=lambda extractor, metrics: SubjectAwareEvaluationPipeline(
        extractor=extractor,
        metrics=metrics,
        subject_field="subject",
    ),
)

built = builder.build(definition, storage_dir=".cache/subject-aware")
report = built.orchestrator.run(dataset=dataset, run_id="subject-run")

# Access subject breakdown
pipeline = built.evaluation_pipeline
if isinstance(pipeline, SubjectAwareEvaluationPipeline):
    print("\nPer-subject accuracy:")
    for subject, accuracy in pipeline.subject_breakdown.items():
        print(f"  {subject}: {accuracy:.2%}")
```

**Output example:**
```
Overall accuracy: 0.725

Per-subject accuracy:
  algebra: 0.850
  geometry: 0.680
  calculus: 0.640
```

## Example 3: Instrumented Provider Router

### Problem

When debugging generation issues, you need visibility into:

- Which models are being called
- Call patterns and ordering
- Provider-specific issues
- Performance bottlenecks

### Solution: TrackingProviderRouter

This router logs all generation calls:

```python
from themis.generation.router import ProviderRouter
from themis.core import entities as core_entities

class TrackingProviderRouter(ProviderRouter):
    """Provider router that tracks all generation calls."""
    
    def __init__(self, providers):
        super().__init__(providers)
        self.call_history = []
        self.call_count_by_model = {}
    
    def generate(self, task: core_entities.GenerationTask):
        # Record the call
        subject = task.metadata.get("subject", "unknown")
        model = task.model.identifier
        call_signature = f"{subject}::{model}"
        
        self.call_history.append(call_signature)
        self.call_count_by_model[model] = (
            self.call_count_by_model.get(model, 0) + 1
        )
        
        # Delegate to parent
        return super().generate(task)
    
    def print_stats(self):
        """Print call statistics."""
        print(f"\nTotal calls: {len(self.call_history)}")
        print("\nCalls by model:")
        for model, count in sorted(self.call_count_by_model.items()):
            print(f"  {model}: {count}")
        
        print("\nCall pattern (first 10):")
        for call in self.call_history[:10]:
            print(f"  {call}")
```

### Usage

```python
# Manually construct experiment with tracking
from themis.generation.router import ProviderRouter
from .generation import TrackingProviderRouter

# Create tracking router
tracking_router = TrackingProviderRouter(providers={
    "fake": fake_provider,
    "openai-compatible": openai_provider,
})

# Create custom runner with tracking router
runner = GenerationRunner(provider=tracking_router)

# Run experiment
# ... (run as normal) ...

# Print tracking info
tracking_router.print_stats()
```

## Example 4: Agentic Workflow with Multi-Step Generation

### Problem

Complex tasks often require multiple generation steps:

1. Plan the approach
2. Execute the plan
3. Verify the result
4. Refine if needed

### Solution: AgenticRunner

This runner implements a plan-then-execute pattern:

```python
from themis.generation.runner import GenerationRunner
from themis.core import entities as core_entities

class AgenticRunner(GenerationRunner):
    """Multi-step generation: plan → execute → answer."""
    
    def __init__(
        self,
        *,
        provider,
        planner_prompt: str,
        final_prompt_prefix: str,
        **kwargs,
    ):
        super().__init__(provider=provider, **kwargs)
        self._planner_prompt = planner_prompt
        self._final_prompt_prefix = final_prompt_prefix
    
    def _generate_single(self, task: core_entities.GenerationTask):
        # Step 1: Generate a plan
        plan_task = core_entities.GenerationTask(
            prompt=self._planner_prompt.format(problem=task.prompt),
            model=task.model,
            sampling=task.sampling,
            metadata={**task.metadata, "step": "planning"},
        )
        
        plan_record = super()._generate_single(plan_task)
        plan_text = plan_record.response.content
        
        # Step 2: Generate final answer using the plan
        final_prompt = self._final_prompt_prefix.format(
            problem=task.prompt,
            plan=plan_text,
        )
        
        final_task = core_entities.GenerationTask(
            prompt=final_prompt,
            model=task.model,
            sampling=task.sampling,
            metadata={
                **task.metadata,
                "step": "execution",
                "plan": plan_text,
            },
        )
        
        final_record = super()._generate_single(final_task)
        
        # Return final record with plan in metadata
        return core_entities.GenerationRecord(
            task=task,  # Original task
            response=final_record.response,
            metadata={
                **final_record.metadata,
                "plan": plan_text,
                "plan_tokens": plan_record.response.usage.total_tokens,
            },
            failures=[],
        )
```

### Usage

```python
from .agentic import AgenticRunner

planner_prompt = """
Given this problem, create a step-by-step plan to solve it:
{problem}

Plan (3-5 steps):
"""

executor_prompt = """
Problem: {problem}

Plan: {plan}

Now execute this plan and provide the final answer as JSON with "answer" field:
"""

# Create builder with agentic runner
builder = ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[metrics.ExactMatch()],
    runner_factory=lambda provider: AgenticRunner(
        provider=provider,
        planner_prompt=planner_prompt,
        final_prompt_prefix=executor_prompt,
    ),
)

built = builder.build(definition, storage_dir=".cache/agentic")
report = built.orchestrator.run(dataset=dataset, run_id="agentic-run")
```

**Benefits:**
- Breaks complex problems into manageable steps
- More interpretable: you can inspect the plan
- Better accuracy on multi-step reasoning
- Useful for debugging model reasoning

## Example 5: Custom Metric

### Problem

You need domain-specific evaluation that goes beyond exact match:

- Partial credit for close answers
- Custom domain logic (e.g., mathematical equivalence)
- Multi-faceted evaluation

### Solution: Custom Metric

```python
from themis.evaluation.metrics import Metric
from themis.core import entities as core_entities

class PartialCreditMetric(Metric):
    """Gives partial credit for answers that are close to correct."""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
    
    def name(self) -> str:
        return f"PartialCredit(tol={self.tolerance})"
    
    def compute(
        self,
        predicted: str,
        reference: str,
        metadata: dict | None = None,
    ) -> core_entities.MetricScore:
        try:
            pred_val = float(predicted.strip())
            ref_val = float(reference.strip())
            
            # Exact match: full credit
            if pred_val == ref_val:
                score = 1.0
            # Within tolerance: partial credit
            elif abs(pred_val - ref_val) <= abs(ref_val * self.tolerance):
                score = 0.5
            else:
                score = 0.0
        except (ValueError, AttributeError):
            # Non-numeric: fall back to string match
            score = 1.0 if predicted.strip() == reference.strip() else 0.0
        
        return core_entities.MetricScore(
            metric_name=self.name(),
            value=score,
            metadata=metadata or {},
            details={
                "predicted": predicted,
                "reference": reference,
            },
        )
```

### Usage

```python
from .metrics import PartialCreditMetric

builder = ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[
        metrics.ExactMatch(),
        PartialCreditMetric(tolerance=0.1),  # 10% tolerance
    ],
)

built = builder.build(definition, storage_dir=".cache/partial-credit")
report = built.orchestrator.run(dataset=dataset, run_id="partial-run")

# View both metrics
print(f"Exact Match: {report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
print(f"Partial Credit: {report.evaluation_report.metrics['PartialCredit(tol=0.1)'].mean:.2%}")
```

## Running the Advanced Example

This example includes a pre-built CLI with several customizations:

```bash
# Run with default settings
uv run python -m experiments.05_advanced.cli run

# Enable subject breakdown
uv run python -m experiments.05_advanced.cli run --enable-subject-breakdown

# Use chain-of-thought prompting
uv run python -m experiments.05_advanced.cli run --prompt-style cot

# Combine options
uv run python -m experiments.05_advanced.cli run \
  --prompt-style cot \
  --enable-subject-breakdown \
  --config-path experiments/05_advanced/config.sample.json \
  --storage-dir .cache/advanced-demo

# Dry run to see configuration
uv run python -m experiments.05_advanced.cli run --dry-run
```

## Configuration Options

The advanced example config supports additional fields:

```json
{
  "run_id": "advanced-demo",
  "storage_dir": ".cache/advanced-demo",
  "resume": true,
  "enable_subject_breakdown": true,
  "prompt_style": "cot",
  "priority_field": "subject",
  "chunk_size": 4,
  "models": [...],
  "samplings": [...],
  "datasets": [...]
}
```

**New fields:**
- `enable_subject_breakdown`: Enable per-subject metrics
- `prompt_style`: "zero-shot" or "cot" (chain-of-thought)
- `priority_field`: Field to prioritize in custom runner
- `chunk_size`: Batch size for prioritized runner

## Best Practices

### 1. Extend, Don't Replace

Prefer extending existing classes over complete rewrites:

```python
# Good: Extend existing functionality
class MyRunner(GenerationRunner):
    def run(self, tasks):
        # Add preprocessing
        tasks = self._preprocess(tasks)
        # Delegate to parent
        return super().run(tasks)

# Avoid: Complete rewrite loses benefits of base implementation
```

### 2. Preserve Interfaces

Keep the same interface as the base class:

```python
# Good: Same signature
def run(self, tasks: Iterable[GenerationTask]) -> Iterator[GenerationRecord]:
    ...

# Avoid: Changing signatures breaks compatibility
```

### 3. Test Custom Components

Write unit tests for custom components:

```python
def test_prioritized_runner():
    # Create test tasks
    tasks = [...]
    
    runner = PrioritizedGenerationRunner(provider=fake_provider)
    records = list(runner.run(tasks))
    
    # Verify ordering
    subjects = [r.task.metadata["subject"] for r in records]
    assert subjects == sorted(subjects)
```

### 4. Document Customizations

Explain why you're customizing and what behavior changes:

```python
class MyCustomRunner(GenerationRunner):
    """
    Custom runner that batches by subject to reduce context switching.
    
    This is useful for models that maintain context across requests,
    as processing similar problems together improves cache hit rates.
    
    Args:
        priority_field: Metadata field to use for batching (default: "subject")
        chunk_size: Maximum batch size (default: 4)
    """
```

### 5. Make Customizations Optional

Use factory patterns to make customizations opt-in:

```python
def create_runner(custom: bool = False):
    if custom:
        return PrioritizedGenerationRunner(...)
    return GenerationRunner(...)
```

## Common Customization Patterns

### Pattern 1: Preprocessing/Postprocessing

Wrap default behavior with pre/post hooks:

```python
class PreprocessingRunner(GenerationRunner):
    def run(self, tasks):
        tasks = self._preprocess(tasks)
        for record in super().run(tasks):
            yield self._postprocess(record)
```

### Pattern 2: Decorator

Add functionality without modifying core logic:

```python
class TimingRunner(GenerationRunner):
    def _generate_single(self, task):
        start = time.time()
        record = super()._generate_single(task)
        duration = time.time() - start
        record.metadata["duration_seconds"] = duration
        return record
```

### Pattern 3: Composition

Combine multiple customizations:

```python
runner = TimingRunner(
    provider=TrackingProviderRouter(
        providers=...
    ),
)
```

## Troubleshooting

**Q: My custom runner isn't being used**
A: Ensure you're passing `runner_factory` to `ExperimentBuilder` or creating the runner manually.

**Q: Custom metrics aren't showing up in reports**
A: Verify the metric is added to the `metrics` list in the builder.

**Q: Subject breakdown is empty**
A: Check that your dataset includes the subject field in metadata.

**Q: How do I debug custom components?**
A: Add logging statements and run with `--log-level debug`.

## Next Steps

- Study the source code in `generation.py`, `pipeline.py`, and `experiment.py`
- Check `themis.generation.runner` for the base runner API
- Review `themis.evaluation.pipeline` for pipeline hooks
- Explore `tests/` for examples of testing custom components

## File Structure

```
05_advanced/
├── README.md              # This file
├── cli.py                 # CLI with customization options
├── config.py              # Extended configuration
├── config.sample.json     # Sample configuration
├── datasets.py            # Dataset loaders
├── experiment.py          # Wiring with custom components
├── generation.py          # Custom runners and routers
└── pipeline.py            # Custom evaluation pipelines
```

## Real-World Use Cases

**Use Case 1: Research Paper**
- Multiple prompt variations (zero-shot, few-shot, CoT)
- Per-category metrics for analysis
- Reproducible configurations

**Use Case 2: Model Debugging**
- Instrumented routers to track calls
- Timing and profiling information
- Detailed failure analysis

**Use Case 3: Production Monitoring**
- Custom metrics for business logic
- Subject/category breakdowns for reporting
- Graceful error handling

**Use Case 4: Advanced Reasoning**
- Multi-step agentic workflows
- Verification and refinement loops
- Interpretable intermediate outputs