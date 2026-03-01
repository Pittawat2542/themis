# Themis Cookbook - Quick Reference

This is a quick reference guide to the Themis examples cookbook. For detailed runnable code, see the `examples/` directory.

## 🎯 Which Example Should I Start With?

| Your Goal | Start Here | Time |
|-----------|------------|------|
| I'm completely new to Themis | [`examples/01_quickstart.py`](examples/01_quickstart.py) | 5 min |
| I want to evaluate a custom dataset inline | [`examples/02_custom_dataset.py`](examples/02_custom_dataset.py) | 5 min |
| I want to scale up execution (workers/retries) | [`examples/03_distributed.py`](examples/03_distributed.py) | 10 min |
| I want to compare two models statistically | [`examples/04_comparison.py`](examples/04_comparison.py) | 15 min |
| I want to interact with Themis via a REST/WebSocket API | [`examples/05_api_server.py`](examples/05_api_server.py) | 15 min |
| I want to write a custom evaluation metric | [`examples/06_custom_metrics.py`](examples/06_custom_metrics.py) | 15 min |
| I want to create a custom storage backend | [`examples/13_custom_storage.py`](examples/13_custom_storage.py) | 15 min |
| I want to build a multi-turn agent | [`examples/14_agentic_evaluation.py`](examples/14_agentic_evaluation.py) | 20 min |
| I want to run a complex, multi-stage R&D pipeline | [`examples/countdown/`](examples/countdown/) | 60 min |


## 🚀 Quick Recipes

These recipes highlight common `themis.evaluate()` patterns.

### 1. Basic Generation with a Hosted API

Themis uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood. You can pass API keys via the environment or directly via `**kwargs`.

```python
import os
import themis

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-..."

report = themis.evaluate(
    "gsm8k",                 # Built-in benchmark preset
    model="openai/gpt-4o",   # LiteLLM provider string
    limit=50,                # Only evaluate the first 50 samples
    temperature=0.0,         # Greedy decoding
)

print(f"Accuracy: {report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
```

### 2. Evaluating a Custom Dataset

You don't need a built-in benchmark to use Themis. You can pass a list of dictionaries directly.

```python
import themis

my_dataset = [
    {"question": "What is the capital of France?", "expected_answer": "Paris"},
    {"question": "Who wrote Hamlet?", "expected_answer": "William Shakespeare"},
]

report = themis.evaluate(
    dataset=my_dataset,
    model="anthropic/claude-3-5-sonnet-20241022",
    prompt="Answer the following question concisely: {question}",
    reference_field="expected_answer", # Tell Themis where the answer is
    metrics=["exact_match"],           # Use the built-in exact match metric
)
```

### 3. Distributed Execution & Rate Limit Resilience

When evaluating against public APIs, you often hit rate limits. Themis has native, multi-threaded retry logic.

```python
import themis

report = themis.evaluate(
    "math500",
    model="ollama/llama3",
    workers=16,          # Number of concurrent Generation requests
    max_retries=5,       # How many times to retry a failed generation (e.g. HTTP 429)
    timeout=60,          # LiteLLM kwargs are passed through directly
)
```

> **Note**: If a generation fails `max_retries` times, Themis does not crash the entire run. It records a generation error for that specific item, returns an empty string, and metrics like `ExactMatch` will score it as `0.0`.

### 4. Resuming from a Cache

Evaluations can be expensive and time-consuming. You can cache and resume them seamlessly using a specific `run_id`.

```python
import themis

# Run 1: Imagine this crashes or you stop it halfway through
report_partial = themis.evaluate(
    "demo",
    model="fake:fake-math-llm",
    run_id="my-expensive-run",     # Unique ID for caching
    storage=".cache/experiments",  # Local storage directory
    resume=True,                   # Enable caching/resumption
)

# Run 2: Running the exact same code will instantly load all completed
# generations from disk and only execute the missing ones.
report_completed = themis.evaluate(
    "demo",
    model="fake:fake-math-llm",
    run_id="my-expensive-run",
    storage=".cache/experiments",
    resume=True,
)
```

### 5. Running the API Web Server

Themis ships with a built-in FastAPI server that provides a REST UI and WebSocket streaming for evaluations.

```bash
uv run themis serve --storage .cache/experiments --port 8000
```
Open `http://localhost:8000` in your browser to view historic runs and trigger new ones.

## 🐛 Troubleshooting

### "No module named X"
Always run your scripts within the `uv` environment:
```bash
uv run python my_script.py
```

### Rate Limits or Connection Timeouts
Reduce your worker count or increase your retries.
```python
themis.evaluate(..., workers=2, max_retries=10)
```

### Results not updating when I change the prompt/code
Themis caching (`resume=True`) uses the `run_id`. If you change your prompt or code, you must either:
1. Change the `run_id`.
2. Delete the `.cache/experiments/{run_id}` folder.
3. Pass `resume=False` to force a complete re-run.

## 📚 Advanced Learning Path

Once you are comfortable with `themis.evaluate()`, check out the [`countdown/`](examples/countdown/) directory.

It is a dense, multi-part internal tutorial showcasing advanced R&D pipelines, SLURM orchestration, dataset synthesis, and reproducibility gates using the programmatic `ExperimentSession` API.