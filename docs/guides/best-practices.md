# Best Practices

Guidelines for effective LLM evaluation with Themis.

## Evaluation Best Practices

### 1. Start Small, Scale Up

Always test with a small sample first:

```python
# Step 1: Test (fast, cheap)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Step 2: Verify (medium sample)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# Step 3: Full run (after verification)
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

**Why**: Catch errors early without wasting API costs.

### 2. Use Meaningful Run IDs

Create descriptive, timestamp-based run IDs:

```python
# Good
run_id = "gsm8k-gpt4-temp07-cot-2024-01-15"

# Bad
run_id = "experiment123"
```

**Format**: `{benchmark}-{model}-{variant}-{date}`

**Why**: Makes tracking and comparing experiments easier.

### 3. Enable Caching

Always use caching for resumability:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,  # Default: True
)
```

**Why**: Resume failed runs without losing progress or money.

### 4. Use Deterministic Settings

For reproducibility, use `temperature=0`:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.0,  # Deterministic
    seed=42,          # If supported by provider
)
```

**Why**: Reproducible results for scientific rigor.

### 5. Monitor Costs

Check costs before scaling:

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)
estimated_full = result.cost * 850  # GSM8K ≈ 8500 samples
print(f"Estimated full cost: ${estimated_full:.2f}")

if estimated_full < 100:  # Budget check
    result = evaluate(benchmark="gsm8k", model="gpt-4")
```

**Why**: Avoid surprising API bills.

---

## Comparison Best Practices

### 1. Use Statistical Tests

Don't rely on raw differences:

```python
# Bad: Manual comparison
accuracy_a = 0.85
accuracy_b = 0.83
diff = accuracy_a - accuracy_b  # 0.02 - is this significant?

# Good: Statistical test
report = compare_runs(
    ["run-a", "run-b"],
    storage_path=".cache",
    statistical_test="bootstrap",
)

if report.pairwise_results[0].is_significant():
    print("Difference is statistically significant")
```

**Why**: Distinguish real improvements from random noise.

### 2. Use Same Test Set

Compare models on identical samples:

```python
# Evaluate both on same data
result_a = evaluate(benchmark="gsm8k", model="gpt-4", run_id="run-a")
result_b = evaluate(benchmark="gsm8k", model="claude-3", run_id="run-b")

# Compare (uses paired test by default)
report = compare_runs(["run-a", "run-b"], storage_path=".cache")
```

**Why**: Paired comparisons have higher statistical power.

### 3. Report Multiple Metrics

Don't rely on a single metric:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "MathVerify", "BLEU"],
)
```

**Why**: Different metrics capture different aspects of performance.

### 4. Consider Practical Significance

A statistically significant difference may not be meaningful:

```python
for result in report.pairwise_results:
    if result.is_significant():
        if abs(result.delta_percent) > 5.0:  # >5% difference
            print(f"✓ {result.metric_name}: Meaningful improvement")
        else:
            print(f"  {result.metric_name}: Significant but small")
```

**Why**: Statistical significance ≠ practical importance.

### 5. Use Appropriate Sample Sizes

Ensure sufficient data for statistical power:

- **n < 10**: Results unreliable
- **10 ≤ n < 30**: Use bootstrap or permutation
- **n ≥ 30**: Any test works
- **n ≥ 100**: High power

```python
# Use appropriate sample size
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)  # Good
```

**Why**: Small samples lack statistical power.

---

## Organization Best Practices

### 1. Structured Storage

Organize experiments logically:

```bash
.cache/experiments/
├── gsm8k-gpt4-baseline-2024-01-15/
├── gsm8k-gpt4-cot-2024-01-15/
├── gsm8k-claude-baseline-2024-01-15/
└── math500-gpt4-baseline-2024-01-16/
```

**Pattern**: `{benchmark}-{model}-{variant}-{date}/`

### 2. Document Experiments

Keep notes about your experiments:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="gsm8k-gpt4-experiment-notes",
    # Add notes in run_id or use external notes file
)
```

Create a `EXPERIMENTS.md`:
```markdown
# Experiments Log

## 2024-01-15: Baseline GSM8K
- Run ID: gsm8k-gpt4-baseline-2024-01-15
- Goal: Establish baseline performance
- Result: 92% accuracy
- Notes: Standard prompt works well
```

### 3. Version Control Results

Commit important results:

```bash
git add .cache/experiments/important-run/
git commit -m "Add baseline GSM8K results"
```

Or export to version control:
```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results/gsm8k-baseline.json",
)
```

---

## Prompt Engineering Best Practices

### 1. Systematic Exploration

Test prompts systematically:

```python
prompts = {
    "zero-shot": "Solve: {prompt}",
    "cot": "Let's think step by step to solve: {prompt}",
    "few-shot": "Example 1: ...\nExample 2: ...\nQ: {prompt}\nA:",
}

for name, template in prompts.items():
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        prompt=template,
        limit=100,
        run_id=f"prompt-{name}",
    )
```

**Why**: Find the best prompt objectively.

### 2. Keep Prompts Simple

Start with simple prompts, add complexity as needed:

```python
# Start simple
prompt_v1 = "{prompt}"

# Add instruction if needed
prompt_v2 = "Solve this problem: {prompt}"

# Add structure if needed
prompt_v3 = "Problem: {prompt}\nLet's solve step by step.\nAnswer:"
```

**Why**: Simpler prompts are more robust.

### 3. Test Prompt Variants

A/B test different wordings:

```python
variants = [
    "Solve: {prompt}",
    "Calculate: {prompt}",
    "Find the answer to: {prompt}",
]

# Test all
for i, variant in enumerate(variants):
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        prompt=variant,
        limit=100,
        run_id=f"variant-{i}",
    )

# Compare
report = compare_runs(
    [f"variant-{i}" for i in range(len(variants))],
    storage_path=".cache",
)
```

**Why**: Small wording changes can affect performance.

---

## Model Comparison Best Practices

### 1. Control Variables

Change only one variable at a time:

```python
# Good: Same prompt, different models
evaluate(benchmark="gsm8k", model="gpt-4", prompt=PROMPT, run_id="gpt4")
evaluate(benchmark="gsm8k", model="claude-3", prompt=PROMPT, run_id="claude")

# Bad: Different everything
evaluate(benchmark="gsm8k", model="gpt-4", prompt=PROMPT_A, run_id="gpt4")
evaluate(benchmark="math500", model="claude-3", prompt=PROMPT_B, run_id="claude")
```

**Why**: Isolate what causes performance differences.

### 2. Use Same Sample Size

Compare on equal-sized samples:

```python
LIMIT = 100  # Use same limit for all

evaluate(benchmark="gsm8k", model="gpt-4", limit=LIMIT, run_id="gpt4")
evaluate(benchmark="gsm8k", model="claude-3", limit=LIMIT, run_id="claude")
```

**Why**: Fair comparison.

### 3. Multiple Benchmarks

Evaluate on multiple benchmarks:

```python
models = ["gpt-4", "claude-3", "gemini-pro"]
benchmarks = ["gsm8k", "math500", "aime24"]

for model in models:
    for benchmark in benchmarks:
        result = evaluate(
            benchmark=benchmark,
            model=model,
            run_id=f"{benchmark}-{model}",
        )
```

**Why**: Models perform differently across domains.

---

## Performance Best Practices

### 1. Appropriate Worker Count

Balance speed vs rate limits:

```python
# API models (respect rate limits)
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=8)

# Local models (can use more workers)
result = evaluate(benchmark="gsm8k", model="ollama/llama3", workers=32)
```

**Why**: Avoid rate limit errors and throttling.

### 2. Use Local Models for Development

Test with local models during development:

```python
# Development: Fast and free
result = evaluate(benchmark="gsm8k", model="fake-math-llm", limit=10)

# Production: Real model
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

**Why**: Save time and money during development.

### 3. Batch Related Experiments

Run related experiments together:

```python
temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]

# Run all at once
for temp in temperatures:
    evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        temperature=temp,
        run_id=f"temp-{temp}",
    )

# Compare after
report = compare_runs(
    [f"temp-{t}" for t in temperatures],
    storage_path=".cache",
)
```

**Why**: Efficient use of time and resources.

---

## Code Quality Best Practices

### 1. Use Type Hints

Leverage Python's type system:

```python
from themis import evaluate
from themis.core.entities import ExperimentReport

def run_evaluation(benchmark: str, model: str) -> ExperimentReport:
    return evaluate(benchmark=benchmark, model=model)
```

**Why**: Catch errors early with type checkers.

### 2. Handle Errors

Wrap evaluations in error handling:

```python
try:
    result = evaluate(benchmark="gsm8k", model="gpt-4")
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"Storage error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log for debugging
```

**Why**: Graceful failure and debugging.

### 3. Use Configuration Objects

For complex configurations:

```python
from dataclasses import dataclass

@dataclass
class EvalConfig:
    benchmark: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    workers: int = 8

config = EvalConfig(benchmark="gsm8k", model="gpt-4")
result = evaluate(**config.__dict__)
```

**Why**: Reusable, testable configurations.

---

## Research Best Practices

### 1. Report Full Details

Include all relevant information:

```python
# Document configuration
config = {
    "benchmark": "gsm8k",
    "model": "gpt-4",
    "temperature": 0.0,
    "max_tokens": 512,
    "prompt": "Problem: {prompt}\nSolution:",
}

result = evaluate(**config)

# Save config with results
import json
output = {
    "config": config,
    "results": result.metrics,
    "num_samples": result.num_samples,
}

with open("experiment_report.json", "w") as f:
    json.dump(output, f, indent=2)
```

### 2. Use Seeds for Reproducibility

When temperature > 0, set seeds:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,
    seed=42,  # For reproducibility
)
```

**Note**: Not all providers support seeds.

### 3. Report Statistical Significance

Always include statistical tests:

```python
from themis.comparison import compare_runs

report = compare_runs(
    ["method-a", "method-b"],
    storage_path=".cache",
)

# Report p-values and effect sizes
for result in report.pairwise_results:
    if result.test_result:
        print(f"{result.metric_name}:")
        print(f"  p-value: {result.test_result.p_value:.4f}")
        print(f"  effect size: {result.test_result.effect_size:.3f}")
```

### 4. Multiple Runs for Robustness

Run multiple times with different seeds:

```python
for seed in [42, 43, 44, 45, 46]:
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        temperature=0.7,
        seed=seed,
        run_id=f"gsm8k-seed{seed}",
    )

# Analyze variance across seeds
```

**Why**: Measure stability of results.

---

## Cost Optimization

### 1. Use Limits for Testing

Start with small limits:

```python
# Test: 10 samples (~$0.50)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Validation: 100 samples (~$5)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# Full: All samples (~$50)
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 2. Use Cheaper Models First

Test with cheaper models:

```python
# Step 1: Test with GPT-3.5 ($0.001/1K tokens)
result = evaluate(benchmark="gsm8k", model="gpt-3.5-turbo")

# Step 2: Confirm with GPT-4 ($0.03/1K tokens)
if result.metrics["ExactMatch"] > 0.5:
    result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 3. Cache Aggressively

Never re-run unnecessarily:

```python
# First run
result = evaluate(benchmark="gsm8k", model="gpt-4", run_id="baseline")

# Later runs use cache automatically
result = evaluate(benchmark="gsm8k", model="gpt-4", run_id="baseline", resume=True)
```

### 4. Use Local Models

For development, use local models:

```python
# Free local inference
result = evaluate(benchmark="gsm8k", model="ollama/llama3")

# Or fake model for testing
result = evaluate(benchmark="demo", model="fake-math-llm")
```

---

## Security Best Practices

### 1. Never Commit API Keys

```bash
# Use environment variables
export OPENAI_API_KEY="sk-..."

# Or .env file (add to .gitignore)
echo "OPENAI_API_KEY=sk-..." > .env
```

Add to `.gitignore`:
```
.env
*.key
credentials.json
```

### 2. Use Service Accounts

For production, use service accounts:

```python
# AWS Bedrock with IAM role
result = evaluate(
    benchmark="gsm8k",
    model="bedrock/anthropic.claude-3",
    # Uses IAM role, no hardcoded credentials
)
```

### 3. Sanitize User Inputs

If using user-provided data:

```python
def sanitize_prompt(prompt: str) -> str:
    # Remove potentially harmful content
    # Validate length
    # Escape special characters
    return cleaned_prompt

dataset = [
    {"prompt": sanitize_prompt(user_input), "answer": expected}
    for user_input, expected in user_data
]

result = evaluate(dataset, model="gpt-4")
```

---

## Team Collaboration

### 1. Shared Storage

Use consistent storage location:

```bash
# Set team-wide storage
export THEMIS_STORAGE="~/shared/themis-experiments"
```

Or use cloud storage (when implemented):
```python
from themis.backends import S3StorageBackend

storage = S3StorageBackend(bucket="team-experiments")
result = evaluate(benchmark="gsm8k", model="gpt-4", storage_backend=storage)
```

### 2. Naming Conventions

Agree on run ID format:

```
{benchmark}-{model}-{user}-{purpose}-{date}
```

Example: `gsm8k-gpt4-alice-baseline-2024-01-15`

### 3. Share Comparison Reports

Export and share comparison results:

```bash
themis compare run-1 run-2 --output comparison.html

# Share via email, Slack, etc.
```

---

## Documentation Best Practices

### 1. Document Your Experiments

Create experiment documentation:

```markdown
# Experiment: GSM8K Baseline
**Date**: 2024-01-15
**Goal**: Establish GPT-4 baseline on GSM8K

## Configuration
- Model: GPT-4
- Temperature: 0.0
- Samples: 8,500
- Prompt: "Problem: {prompt}\nSolution:"

## Results
- ExactMatch: 92.3%
- MathVerify: 94.1%
- Cost: $85

## Findings
- GPT-4 performs well on elementary math
- Few errors are arithmetic mistakes
- Most errors are reading comprehension
```

### 2. Keep Experiment Journal

Track what you've tried:

```markdown
# Experiment Journal

## 2024-01-15: Temperature Study
Tried temperatures: 0.0, 0.3, 0.5, 0.7, 1.0
Result: 0.0 performs best (92% vs 85% at 0.7)
Conclusion: Use deterministic sampling

## 2024-01-16: Prompt Engineering
Tried CoT prompt: "Let's think step by step"
Result: No significant improvement (92% → 93%)
Conclusion: Standard prompt sufficient for GSM8K
```

---

## Common Pitfalls to Avoid

### ❌ Don't: Compare on Different Datasets

```python
# Bad
result_a = evaluate(benchmark="gsm8k", model="gpt-4")
result_b = evaluate(benchmark="math500", model="gpt-4")
# Can't compare - different benchmarks!
```

### ❌ Don't: Ignore Statistical Significance

```python
# Bad
acc_a = 0.85
acc_b = 0.83
print("Model A is better!")  # Maybe just noise?

# Good
report = compare_runs(["run-a", "run-b"], storage_path=".cache")
if report.pairwise_results[0].is_significant():
    print("Model A is significantly better")
```

### ❌ Don't: Use Single Metric

```python
# Bad
result = evaluate(benchmark="gsm8k", model="gpt-4", metrics=["ExactMatch"])

# Good
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "MathVerify", "BLEU"],
)
```

### ❌ Don't: Skip Testing

```python
# Bad: Run full evaluation immediately
result = evaluate(benchmark="gsm8k", model="gpt-4")  # $50+

# Good: Test first
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)  # $0.50
```

### ❌ Don't: Hard-Code API Keys

```python
# Bad
api_key = "sk-..."  # Never do this!

# Good
import os
api_key = os.environ["OPENAI_API_KEY"]
```

---

## Summary

**Key Principles:**

1. **Start small, scale up** - Test before committing resources
2. **Use statistical tests** - Don't trust raw numbers
3. **Be reproducible** - Document everything, use deterministic settings
4. **Monitor costs** - Check before scaling
5. **Multiple metrics** - Don't rely on single number
6. **Organize systematically** - Clear naming, documentation
7. **Collaborate effectively** - Shared storage, conventions
8. **Secure credentials** - Environment variables, never commit

**Remember**: Good evaluation practices lead to reliable, reproducible research!

---

## Further Reading

- [Evaluation Guide](guides/evaluation.md) - Detailed evaluation guide
- [Comparison Guide](COMPARISON.md) - Statistical comparison
- [Configuration Guide](guides/configuration.md) - All configuration options
- [FAQ](FAQ.md) - Common questions
