# Themis Jupyter Tutorials

Interactive tutorials for learning Themis through hands-on examples.

## Prerequisites

```bash
# Install Themis with all features
pip install themis-eval[math,nlp,code,server]

# Install Jupyter
pip install jupyter

# Or use uv
uv pip install themis-eval[math,nlp,code,server] jupyter
```

## Tutorials

### 01. Quick Start (`01_quickstart.ipynb`)

Learn the basics of Themis:
- Running your first evaluation
- Using built-in benchmarks
- Customizing model parameters
- Caching and resuming runs
- Evaluating custom datasets

**Duration**: 10-15 minutes  
**Prerequisites**: None

### 02. Statistical Comparison (`02_comparison.ipynb`)

Compare multiple experiment runs:
- Running multiple experiments
- Statistical significance testing (t-test, bootstrap, permutation)
- Interpreting p-values and effect sizes
- Win/loss matrices for 3+ runs
- Exporting comparison reports

**Duration**: 15-20 minutes  
**Prerequisites**: Tutorial 01

## Running the Tutorials

### Option 1: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory
# Open any .ipynb file
```

### Option 2: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

### Option 3: VS Code

1. Install the Jupyter extension in VS Code
2. Open any `.ipynb` file
3. Select Python kernel
4. Run cells with `Shift+Enter`

## Tips

### Using API Keys

To use real models (GPT-4, Claude, etc.), set your API keys before running:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
```

Or set them in your shell:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
```

### Testing Without API Keys

All tutorials use the `fake-math-llm` model by default, which doesn't require API keys. This is perfect for:
- Learning the Themis API
- Testing your code
- Understanding the workflow

### Storage Location

By default, Themis stores results in `.cache/experiments/`. You can change this:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage="/path/to/custom/storage",
)
```

### Running Experiments Faster

For faster experimentation, use small limits:

```python
# Fast testing (10 samples)
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Full evaluation
result = evaluate(benchmark="gsm8k", model="gpt-4")  # All samples
```

## Next Steps

After completing the tutorials:

1. **Read the Documentation**
   - [Main Documentation](../docs/index.md)
   - [API Server Guide](../docs/API_SERVER.md)
   - [Extending Backends](../docs/EXTENDING_BACKENDS.md)

2. **Try the Examples**
   - Check out [examples-simple/](../examples-simple/)
   - More advanced examples in [examples/](../examples/)

3. **Explore the CLI**
   ```bash
   themis eval --help
   themis compare --help
   themis serve --help
   ```

4. **Join the Community**
   - Report issues on [GitHub Issues](https://github.com/yourusername/themis/issues)
   - Discuss on [GitHub Discussions](https://github.com/yourusername/themis/discussions)

## Common Issues

### Import Error

```python
# If you get: ModuleNotFoundError: No module named 'themis'
# Install Themis first:
!pip install themis-eval
```

### Kernel Crashes

If your kernel crashes:
1. Restart the kernel
2. Reduce the `limit` parameter
3. Reduce `workers` parameter

### API Rate Limits

If you hit rate limits:
- Add delays between requests
- Reduce `workers` parameter
- Use `limit` to evaluate fewer samples

## Contributing

Found an issue or want to add a tutorial?
- Open an issue: [GitHub Issues](https://github.com/yourusername/themis/issues)
- Submit a PR: [GitHub Pull Requests](https://github.com/yourusername/themis/pulls)

## License

MIT License - see [LICENSE](../LICENSE) for details.
