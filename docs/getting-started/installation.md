# Installation

This guide covers different ways to install Themis based on your needs.

## Requirements

- **Python**: 3.12 or higher
- **Operating System**: macOS, Linux, or Windows
- **Package Manager**: pip or uv (recommended)

## Basic Installation

### Using pip

```bash
pip install themis-eval
```

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Themis
uv pip install themis-eval
```

## Installation with Optional Features

Themis has several optional feature groups:

### Math Evaluation

For math problem evaluation (GSM8K, MATH500, etc.):

```bash
pip install themis-eval[math]
```

Includes:
- `datasets` - HuggingFace datasets
- `math-verify` - Symbolic math verification

### NLP Metrics

For natural language processing metrics:

```bash
pip install themis-eval[nlp]
```

Includes:
- `sacrebleu` - BLEU score
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore
- `nltk` - METEOR and other NLP tools

### Code Metrics

For code generation evaluation:

```bash
pip install themis-eval[code]
```

Includes:
- `codebleu` - CodeBLEU metric

### API Server

For running the web dashboard:

```bash
pip install themis-eval[server]
```

Includes:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `websockets` - Real-time updates

### Visualization

For plotting and visualization:

```bash
pip install themis-eval[viz]
```

Includes:
- `plotly` - Interactive plots

### All Features

Install everything:

```bash
pip install themis-eval[all]
```

Or with uv:

```bash
uv pip install themis-eval[all]
```

## Development Installation

For contributing to Themis:

```bash
# Clone the repository
git clone https://github.com/pittawat2542/themis.git
cd themis

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"

# Or with uv
uv pip install -e ".[dev,all]"
```

Development dependencies include:
- `pytest` - Testing framework
- Additional testing and linting tools

## Verify Installation

Check that Themis is installed correctly:

```python
import themis
print(themis.__version__)
```

Or use the CLI:

```bash
themis --version
```

## API Keys

Themis uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM providers. Set up your API keys:

### OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Azure OpenAI

```bash
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2023-05-15"
```

### Other Providers

See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for other providers.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Make sure Themis is installed
pip list | grep themis

# Reinstall if needed
pip install --force-reinstall themis-eval
```

### Version Conflicts

If you have dependency conflicts:

```bash
# Use uv which has better dependency resolution
uv pip install themis-eval

# Or create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install themis-eval
```

### Optional Dependencies

If optional features don't work:

```bash
# Install specific feature group
pip install themis-eval[math,nlp,code]

# Or install all features
pip install themis-eval[all]
```

## Updating

Keep Themis up to date:

```bash
pip install --upgrade themis-eval
```

Or with uv:

```bash
uv pip install --upgrade themis-eval
```

## Uninstallation

To remove Themis:

```bash
pip uninstall themis-eval
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with Themis
- [Core Concepts](concepts.md) - Understand how Themis works
- [Examples](../tutorials/examples.md) - See working code examples
