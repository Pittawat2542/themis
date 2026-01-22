# Jupyter Notebook Tutorials

Interactive tutorials for learning Themis.

## Available Notebooks

### 01. Quick Start

**Path**: `notebooks/01_quickstart.ipynb`

Learn the basics:
- Running your first evaluation
- Using built-in benchmarks
- Customizing model parameters
- Caching and resuming runs
- Evaluating custom datasets

**Duration**: 10-15 minutes  
**Prerequisites**: None

[View Notebook →](../notebooks/01_quickstart.ipynb)

### 02. Statistical Comparison

**Path**: `notebooks/02_comparison.ipynb`

Compare experiment runs:
- Running multiple experiments
- Statistical tests (t-test, bootstrap, permutation)
- Interpreting p-values and effect sizes
- Win/loss matrices for 3+ runs
- Exporting comparison reports

**Duration**: 15-20 minutes  
**Prerequisites**: Tutorial 01

[View Notebook →](../notebooks/02_comparison.ipynb)

---

## Setup

### Installation

```bash
# Install Themis with all features
pip install themis-eval[all]

# Install Jupyter
pip install jupyter

# Or with uv
uv pip install themis-eval[all] jupyter
```

### Running Notebooks

#### Option 1: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory
# Open any .ipynb file
```

#### Option 2: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

#### Option 3: VS Code

1. Install the Jupyter extension
2. Open any `.ipynb` file
3. Select Python kernel
4. Run cells with `Shift+Enter`

#### Option 4: Google Colab

Upload notebooks to Google Colab:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload notebook file
3. Install Themis: `!pip install themis-eval`
4. Run cells

---

## Using API Keys

### Set Environment Variables

For real model evaluations, set API keys:

```python
import os

# OpenAI
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Anthropic
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

# Azure OpenAI
os.environ["AZURE_API_KEY"] = "your-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com"
```

### Testing Without API Keys

All tutorials use `fake-math-llm` by default (no API key needed):

```python
result = evaluate(
    benchmark="demo",
    model="fake-math-llm",
    limit=10,
)
```

---

## Tips and Tricks

### Start Small

Use `limit` parameter for faster testing:

```python
# Quick test (10 samples)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Full evaluation
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### Save Outputs

Export results for later analysis:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.json",
)
```

### Use Descriptive Run IDs

Make it easy to identify experiments:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="gsm8k-gpt4-baseline-2024-01-15",
)
```

### Monitor Progress

Use callbacks for progress tracking:

```python
def log_progress(record):
    print(f"✓ {record.id}")

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    on_result=log_progress,
)
```

---

## Common Issues

### Kernel Crashes

If your kernel crashes:
1. Restart the kernel
2. Reduce `limit` parameter
3. Reduce `workers` parameter
4. Check available memory

### Import Errors

```python
# If you get: ModuleNotFoundError: No module named 'themis'
!pip install themis-eval
```

### API Rate Limits

If you hit rate limits:
- Reduce `workers` parameter
- Add delays between requests
- Use smaller `limit` for testing

### Slow Execution

If evaluation is slow:
- Increase `workers` parameter
- Use smaller `limit` for testing
- Check network connection
- Verify API key is correct

---

## Next Steps

After completing the tutorials:

1. **Read the Documentation**
   - [Evaluation Guide](../guides/evaluation.md)
   - [Comparison Guide](../COMPARISON.md)
   - [API Reference](../api/overview.md)

2. **Try the Examples**
   - Check out [examples-simple/](../../examples-simple/)
   - More advanced examples in [examples/](../../examples/)

3. **Explore the CLI**
   ```bash
   themis eval --help
   themis compare --help
   themis serve --help
   ```

4. **Join the Community**
   - [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)
   - [GitHub Issues](https://github.com/pittawat2542/themis/issues)

---

## Contributing

Want to add a tutorial?
- Fork the repository
- Add your notebook to `notebooks/`
- Update this documentation
- Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
