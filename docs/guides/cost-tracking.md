# Cost Tracking and Estimation

Themis provides comprehensive cost tracking and estimation features to help you monitor and control LLM experiment costs. This guide covers cost estimation, real-time tracking, budget monitoring, and cost analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Cost Estimation](#cost-estimation)
- [Real-Time Cost Tracking](#real-time-cost-tracking)
- [Budget Monitoring](#budget-monitoring)
- [Pricing Information](#pricing-information)
- [Cost Visualization](#cost-visualization)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Quick Start

### Estimate Cost Before Running

```bash
# Estimate cost for GPT-4 on 100 samples
uv run python -m themis.cli estimate-cost \
  --model gpt-4 \
  --dataset-size 100

# With custom token estimates
uv run python -m themis.cli estimate-cost \
  --model claude-3-5-sonnet-20241022 \
  --dataset-size 1000 \
  --avg-prompt-tokens 800 \
  --avg-completion-tokens 400
```

### Check Model Pricing

```bash
# Show pricing for specific model
uv run python -m themis.cli show-pricing --model gpt-4

# List all available models
uv run python -m themis.cli show-pricing --list-all

# Compare costs across models
uv run python -m themis.cli show-pricing \
  --compare-models gpt-4 \
  --compare-models gpt-3.5-turbo \
  --compare-models claude-3-haiku-20240307
```

### Automatic Cost Tracking

Cost tracking is **automatically enabled** for all experiments. Every experiment run tracks:
- Token usage (prompt, completion, total)
- Per-sample costs
- Per-model costs (for multi-model experiments)
- Total generation cost

Results are saved in:
- HTML reports (visual cost breakdown)
- JSON reports (`report.json` metadata)
- Experiment metadata

## Cost Estimation

### CLI Estimation

Use `estimate-cost` to preview experiment costs before running:

```bash
uv run python -m themis.cli estimate-cost \
  --model gpt-4 \
  --dataset-size 500 \
  --avg-prompt-tokens 600 \
  --avg-completion-tokens 350
```

**Output:**
```
================================================================================
Cost Estimate
================================================================================

Model: gpt-4
Dataset size: 500 samples
Avg tokens per sample: 600 prompt + 350 completion

💰 Estimated Cost
  Total: $30.0000
  Per sample: $0.060000
  95% CI: $6.0000 - $54.0000

📊 Breakdown
  Generation: $30.0000
  Evaluation: $0.0000

================================================================================

⚠️  Warning: Estimated cost is $30.00. Consider using --limit for initial testing.
```

### Programmatic Estimation

```python
from themis.experiment.cost import estimate_experiment_cost

# Estimate cost
estimate = estimate_experiment_cost(
    model="gpt-4",
    dataset_size=500,
    avg_prompt_tokens=600,
    avg_completion_tokens=350,
)

print(f"Estimated cost: ${estimate.estimated_cost:.2f}")
print(f"Range: ${estimate.lower_bound:.2f} - ${estimate.upper_bound:.2f}")
print(f"Per sample: ${estimate.assumptions['cost_per_sample']:.6f}")

# Access breakdown
for phase, cost in estimate.breakdown_by_phase.items():
    print(f"{phase}: ${cost:.4f}")
```

### Understanding Estimates

**Confidence Intervals:**
- Estimates include 95% confidence intervals
- Accounts for ~20% variance in token usage
- Actual costs may vary based on prompt complexity

**Token Estimation:**
- Use realistic token counts for your use case
- Run small pilot (5-10 samples) to measure actual usage
- Check HTML reports for actual token counts

## Real-Time Cost Tracking

### Automatic Tracking

Cost tracking is built into the experiment orchestrator:

```python
from themis.experiment.builder import ExperimentBuilder

# Cost tracking happens automatically
experiment = (
    ExperimentBuilder()
    .with_dataset_loader(loader)
    .with_prompt_spec(prompt_spec)
    .with_model_spec(model_spec)
    .with_sampling_spec(sampling_spec)
    .with_runner(runner)
    .with_metric(metric)
    .with_cache(storage_dir=".cache/runs", run_id="my-run")
    .build()
)

report = experiment.run()

# Cost data is in metadata
cost = report.metadata["cost"]
print(f"Total cost: ${cost['total_cost']:.4f}")
print(f"Generation cost: ${cost['generation_cost']:.4f}")
print(f"Token counts: {cost['token_counts']}")
```

### Manual Tracking

For custom workflows, use `CostTracker` directly:

```python
from themis.experiment.cost import CostTracker
from themis.experiment.pricing import calculate_cost

tracker = CostTracker()

# Record each generation
for record in generation_results:
    if record.output and record.output.usage:
        usage = record.output.usage
        cost = calculate_cost(
            model="gpt-4",
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        tracker.record_generation(
            model="gpt-4",
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cost=cost,
        )

# Get breakdown
breakdown = tracker.get_breakdown()
print(f"Total: ${breakdown.total_cost:.4f}")
print(f"Generation: ${breakdown.generation_cost:.4f}")
print(f"API calls: {breakdown.api_calls}")
print(f"Total tokens: {breakdown.token_counts['total_tokens']}")

# Per-model costs
for model, cost in breakdown.per_model_costs.items():
    print(f"{model}: ${cost:.4f}")
```

## Budget Monitoring

### Using BudgetMonitor

Monitor costs during long-running experiments:

```python
from themis.experiment.cost import BudgetMonitor

# Set budget limit
monitor = BudgetMonitor(
    max_cost=10.0,        # $10 maximum
    alert_threshold=0.8   # Alert at 80%
)

# Track costs as they occur
for sample in dataset:
    # ... generate response ...
    cost = calculate_cost(model, prompt_tokens, completion_tokens)

    monitor.add_cost(cost)

    # Check budget status
    within_budget, message = monitor.check_budget()
    if not within_budget:
        print(f"⚠️  {message}")
        break

    # Show warnings
    if "Warning" in message:
        print(f"⚠️  {message}")

    # Check remaining budget
    remaining = monitor.remaining_budget()
    print(f"Remaining: ${remaining:.2f}")

# Final status
percentage = monitor.percentage_used()
print(f"Used {percentage:.1f}% of budget")
```

### Budget Enforcement Pattern

```python
def run_with_budget(dataset, max_cost):
    """Run experiment with strict budget enforcement."""
    monitor = BudgetMonitor(max_cost=max_cost, alert_threshold=0.9)
    tracker = CostTracker()

    results = []
    for i, sample in enumerate(dataset):
        # Check budget before generating
        within_budget, message = monitor.check_budget()
        if not within_budget:
            print(f"Budget exceeded at sample {i}/{len(dataset)}")
            print(message)
            break

        # Generate response
        response = generate(sample)
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        # Update tracking
        monitor.add_cost(cost)
        tracker.record_generation(model, prompt_tokens, completion_tokens, cost)

        results.append(response)

    breakdown = tracker.get_breakdown()
    return results, breakdown
```

## Pricing Information

### Supported Models

Themis includes pricing for 20+ models:

**OpenAI:**
- GPT-4: $30/$60 per 1M tokens
- GPT-4 Turbo: $10/$30 per 1M tokens
- GPT-3.5 Turbo: $0.50/$1.50 per 1M tokens

**Anthropic:**
- Claude 3.5 Sonnet: $3/$15 per 1M tokens
- Claude 3 Opus: $15/$75 per 1M tokens
- Claude 3 Haiku: $0.25/$1.25 per 1M tokens

**Google:**
- Gemini 1.5 Pro: $1.25/$5.00 per 1M tokens
- Gemini 1.5 Flash: $0.07/$0.30 per 1M tokens

**Others:**
- Mistral Large, Cohere Command-R, Llama 3.1, and more

### Viewing Pricing

```bash
# Show all models with pricing
uv run python -m themis.cli show-pricing --list-all

# Compare specific models
uv run python -m themis.cli show-pricing \
  --compare-models gpt-4 \
  --compare-models claude-3-5-sonnet-20241022 \
  --compare-models gemini-1.5-pro
```

**Output:**
```
================================================================================
Cost Comparison (1000 prompt + 500 completion tokens)
================================================================================

  gemini-1.5-pro                           | $0.003750
                                             | ($1.25 / $5.00 per 1M)
  claude-3-5-sonnet-20241022               | $0.010500
                                             | ($3.00 / $15.00 per 1M)
  gpt-4                                    | $0.060000
                                             | ($30.00 / $60.00 per 1M)

Relative costs (vs gemini-1.5-pro):
  claude-3-5-sonnet-20241022               | 2.8x more expensive
  gpt-4                                    | 16.0x more expensive

================================================================================
```

### Programmatic Pricing Access

```python
from themis.experiment.pricing import (
    get_provider_pricing,
    calculate_cost,
    compare_provider_costs,
    get_all_models,
)

# Get pricing for specific model
pricing = get_provider_pricing("gpt-4")
print(f"Prompt: ${pricing['prompt_tokens'] * 1_000_000:.2f}/1M tokens")
print(f"Completion: ${pricing['completion_tokens'] * 1_000_000:.2f}/1M tokens")

# Calculate cost for specific usage
cost = calculate_cost("gpt-4", prompt_tokens=1000, completion_tokens=500)
print(f"Cost: ${cost:.6f}")

# Compare multiple models
models = ["gpt-4", "gpt-3.5-turbo", "claude-3-haiku-20240307"]
costs = compare_provider_costs(
    prompt_tokens=1000,
    completion_tokens=500,
    models=models
)
for model, cost in sorted(costs.items(), key=lambda x: x[1]):
    print(f"{model}: ${cost:.6f}")

# List all available models
all_models = get_all_models()
print(f"Total models: {len(all_models)}")
```

### Model Name Normalization

Themis normalizes model names automatically:

```python
from themis.experiment.pricing import normalize_model_name

# All resolve to canonical names
print(normalize_model_name("openai/gpt-4"))           # gpt-4
print(normalize_model_name("gpt-4-0613"))             # gpt-4
print(normalize_model_name("claude-3-opus"))          # claude-3-opus-20240229
print(normalize_model_name("claude-3.5-sonnet"))      # claude-3-5-sonnet-20241022
```

**Default Pricing:**
- Unknown models use default: $1/$2 per 1M tokens
- Add custom models to `PRICING_TABLE` in `themis/experiment/pricing.py`

## Cost Visualization

### HTML Reports

Cost data is automatically included in HTML reports:

```python
from themis.experiment.export import render_html_report

html = render_html_report(
    report=experiment_report,
    title="My Experiment",
)

# Save to file
with open("report.html", "w") as f:
    f.write(html)
```

**Cost Section Includes:**
- **Total Cost**: Prominently displayed in green
- **Breakdown**: Generation vs Evaluation costs
- **Token Counts**: Prompt, completion, and total tokens
- **Per-Model Costs**: Table with cost distribution

### JSON Reports

Cost data is saved in `report.json`:

```json
{
  "metadata": {
    "cost": {
      "total_cost": 0.045,
      "generation_cost": 0.045,
      "evaluation_cost": 0.0,
      "per_model_costs": {
        "gpt-4": 0.045
      },
      "token_counts": {
        "prompt_tokens": 5000,
        "completion_tokens": 2500,
        "total_tokens": 7500
      },
      "api_calls": 10,
      "currency": "USD"
    }
  }
}
```

### Multi-Experiment Comparison

Use the `compare` command to analyze costs across runs:

```bash
uv run python -m themis.cli compare \
  --run-ids run-1 run-2 run-3 \
  --storage .cache/runs \
  --output comparison.csv
```

The comparison includes cost data for each experiment, making it easy to identify cost-efficient configurations.

## Best Practices

### 1. Estimate Before Running

Always estimate costs before running large experiments:

```bash
# Estimate first
uv run python -m themis.cli estimate-cost \
  --model gpt-4 \
  --dataset-size 1000

# Then run with limit for pilot
uv run python -m themis.cli math500 \
  --limit 10 \
  --model gpt-4
```

### 2. Start with Small Pilots

Use `--limit` to test on small samples first:

```bash
# Pilot run (5 samples)
uv run python -m themis.cli math500 --limit 5

# Check costs in HTML report
# Scale up if costs are acceptable
uv run python -m themis.cli math500 --limit 50
```

### 3. Choose Cost-Effective Models

Use pricing comparison to select models:

```bash
uv run python -m themis.cli show-pricing \
  --compare-models gpt-4 \
  --compare-models gpt-3.5-turbo \
  --compare-models claude-3-haiku-20240307
```

**Cost-Effectiveness Strategies:**
- **Initial exploration**: Use cheaper models (GPT-3.5, Claude Haiku, Gemini Flash)
- **Validation**: Use mid-tier models (Claude Sonnet, GPT-4 Turbo)
- **Final evaluation**: Use premium models (GPT-4, Claude Opus) on best candidates

### 4. Monitor Token Usage

Check HTML reports for actual token usage:

```python
# After running experiment, check report
report = experiment.run()
tokens = report.metadata["cost"]["token_counts"]
print(f"Avg prompt tokens: {tokens['prompt_tokens'] / len(dataset):.0f}")
print(f"Avg completion tokens: {tokens['completion_tokens'] / len(dataset):.0f}")
```

Use actual averages for future cost estimates.

### 5. Use Budget Monitoring

For long-running experiments, implement budget checks:

```python
from themis.experiment.cost import BudgetMonitor

monitor = BudgetMonitor(max_cost=50.0, alert_threshold=0.8)

# Check before each expensive operation
for batch in batches:
    within_budget, message = monitor.check_budget()
    if not within_budget:
        print(f"Stopping: {message}")
        break
    # ... process batch ...
```

### 6. Optimize Prompts for Cost

**Token Reduction Strategies:**
- Use concise prompts (remove unnecessary context)
- Limit `max_tokens` to expected response length
- Use few-shot only when necessary (adds prompt tokens)
- Consider prompt compression techniques

**Example:**
```python
# Expensive: verbose prompt + high max_tokens
prompt = """Please carefully solve the following math problem.
Show all your work and explain your reasoning in detail.
Problem: {problem}
"""
sampling = SamplingSpec(max_tokens=1000)

# Cost-effective: concise prompt + tight max_tokens
prompt = "Solve: {problem}"
sampling = SamplingSpec(max_tokens=300)
```

### 7. Track Costs Across Experiments

Use `compare` to analyze cost trends:

```bash
uv run python -m themis.cli compare \
  --run-ids pilot-1 pilot-2 pilot-3 \
  --output cost_analysis.csv
```

Review `cost_analysis.csv` to identify cost drivers.

### 8. Consider Pareto Optimization

Find optimal accuracy/cost tradeoffs:

```bash
uv run python -m themis.cli pareto \
  --run-ids run-1 run-2 run-3 run-4 \
  --objectives accuracy total_cost \
  --maximize true false
```

Pareto-optimal runs give best accuracy per dollar spent.

## API Reference

### Cost Estimation

#### `estimate_experiment_cost()`

```python
from themis.experiment.cost import estimate_experiment_cost

estimate = estimate_experiment_cost(
    model: str,                        # Model identifier
    dataset_size: int,                 # Number of samples
    avg_prompt_tokens: int = 500,      # Avg prompt tokens per sample
    avg_completion_tokens: int = 300,  # Avg completion tokens per sample
    confidence_level: float = 0.95,    # Confidence level for bounds
) -> CostEstimate
```

**Returns:** `CostEstimate` with:
- `estimated_cost`: Expected cost in USD
- `lower_bound`: Lower 95% confidence bound
- `upper_bound`: Upper 95% confidence bound
- `breakdown_by_phase`: Dict of costs by phase
- `assumptions`: Dict of assumptions used

### Cost Tracking

#### `CostTracker`

```python
from themis.experiment.cost import CostTracker

tracker = CostTracker()

# Record generation
tracker.record_generation(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
)

# Record evaluation (if LLM-based)
tracker.record_evaluation(
    metric: str,
    cost: float,
)

# Get breakdown
breakdown: CostBreakdown = tracker.get_breakdown()

# Reset tracker
tracker.reset()
```

#### `CostBreakdown`

```python
@dataclass
class CostBreakdown:
    total_cost: float                  # Total cost in USD
    generation_cost: float             # Cost of generation
    evaluation_cost: float             # Cost of evaluation
    per_sample_costs: list[float]      # Cost per sample
    per_model_costs: dict[str, float]  # Cost by model
    token_counts: dict[str, int]       # Token usage stats
    api_calls: int                     # Total API calls
    currency: str = "USD"              # Currency code
```

### Budget Monitoring

#### `BudgetMonitor`

```python
from themis.experiment.cost import BudgetMonitor

monitor = BudgetMonitor(
    max_cost: float,              # Maximum allowed cost
    alert_threshold: float = 0.8, # Alert threshold (0.0-1.0)
)

# Add cost
monitor.add_cost(cost: float)

# Check budget
within_budget: bool, message: str = monitor.check_budget()

# Get remaining budget
remaining: float = monitor.remaining_budget()

# Get percentage used
percentage: float = monitor.percentage_used()
```

### Pricing

#### `calculate_cost()`

```python
from themis.experiment.pricing import calculate_cost

cost: float = calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, float] | None = None,  # Optional custom pricing
)
```

#### `get_provider_pricing()`

```python
from themis.experiment.pricing import get_provider_pricing

pricing: dict[str, float] = get_provider_pricing(model: str)
# Returns: {"prompt_tokens": 0.00003, "completion_tokens": 0.00006}
```

#### `compare_provider_costs()`

```python
from themis.experiment.pricing import compare_provider_costs

costs: dict[str, float] = compare_provider_costs(
    prompt_tokens: int,
    completion_tokens: int,
    models: list[str],
)
# Returns: {"gpt-4": 0.06, "gpt-3.5-turbo": 0.00125, ...}
```

#### `get_all_models()`

```python
from themis.experiment.pricing import get_all_models

models: list[str] = get_all_models()
# Returns: ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-20241022", ...]
```

## Troubleshooting

### Missing Cost Data

**Problem:** HTML report doesn't show cost section

**Solution:**
- Check that provider returns usage data in `ModelOutput.usage`
- LiteLLM provider automatically populates usage data
- Custom providers must populate `usage` field:

```python
usage_dict = {
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "total_tokens": total_tokens,
}

return ModelOutput(text=text, raw=raw, usage=usage_dict)
```

### Inaccurate Estimates

**Problem:** Actual costs differ significantly from estimates

**Solution:**
- Run pilot (5-10 samples) to measure actual token usage
- Update `avg_prompt_tokens` and `avg_completion_tokens` based on pilot
- Re-run estimation with actual averages

```bash
# Step 1: Run pilot
uv run python -m themis.cli math500 --limit 5

# Step 2: Check HTML report for average tokens
# Step 3: Re-estimate with actual values
uv run python -m themis.cli estimate-cost \
  --model gpt-4 \
  --dataset-size 500 \
  --avg-prompt-tokens 650 \
  --avg-completion-tokens 425
```

### Unknown Model Pricing

**Problem:** Warning: "Using default pricing for unknown model"

**Solution:**
- Add model to `PRICING_TABLE` in `themis/experiment/pricing.py`
- Or provide custom pricing:

```python
from themis.experiment.pricing import calculate_cost

custom_pricing = {
    "prompt_tokens": 0.00001,     # $10 per 1M tokens
    "completion_tokens": 0.00002, # $20 per 1M tokens
}

cost = calculate_cost(
    "my-custom-model",
    prompt_tokens=1000,
    completion_tokens=500,
    pricing=custom_pricing
)
```

## Examples

See also:
- `MULTI_EXPERIMENT_COMPARISON.md` - Compare costs across experiments
- `IMPROVEMENT_PLAN.md` - Roadmap for cost features
- `examples/getting_started/` - Basic cost tracking examples

## Updating Pricing Data

Pricing is current as of **November 2024**. To update:

1. Edit `themis/experiment/pricing.py`
2. Update `PRICING_TABLE` with new rates
3. Run tests: `uv run pytest tests/experiment/test_pricing.py`
4. Update this documentation

**Sources:**
- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://www.anthropic.com/pricing
- Google: https://ai.google.dev/pricing
- Others: Check provider websites
