# Prompt Engineering Example - Complete Guide

This example demonstrates how to use Themis for systematic prompt engineering experiments, testing multiple prompt variations on various models and benchmarks with standard metrics.

## Key Features Demonstrated

1. **Multiple Prompt Variations**: Define and test different prompting strategies (zero-shot, few-shot, chain-of-thought)
2. **Systematic Comparison**: Compare prompt effectiveness across different models and datasets
3. **Standard Metrics**: Use built-in accuracy metrics like ExactMatch
4. **Export Capabilities**: Export results to CSV, JSON, and HTML for analysis
5. **Analysis Tools**: Built-in analysis to compare prompt strategy effectiveness

## Running with Local LLM

The example is pre-configured to work with a local LLM running on `http://localhost:1234/v1`, such as those served by LM Studio, Ollama, or vLLM.

### Prerequisites

1. Ensure your local LLM server is running on `http://localhost:1234/v1`
2. Verify the server is accessible with a test request:
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-vl-30b",
    "messages": [
      { "role": "user", "content": "Hello!" }
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
  }'
```

### Troubleshooting Common Issues

**If you see "Provider List" messages or all requests fail:**

1. **Verify your server is running**: Test with the curl command above
2. **Check model name**: Ensure the model name in config matches what's loaded in your local server
3. **Port and URL**: Confirm your server is running on `http://localhost:1234/v1`
4. **Model compatibility**: Some local servers may need the model name to match what's actually loaded

The configuration uses these key parameters:
- `"api_base": "http://localhost:1234/v1"` - Points to your local server
- `"custom_llm_provider": "openai"` - Treats local server as OpenAI-compatible
- `"api_key": "not-needed"` - Most local servers ignore this value

### Basic Execution with Local Model
```bash
# Run with local model (default configuration)
uv run python -m examples.prompt_engineering.cli run

# Run with dry-run to see the plan without executing
uv run python -m examples.prompt_engineering.cli run --dry-run
```

### With Analysis
```bash
# Run and show prompt strategy comparison
uv run python -m examples.prompt_engineering.cli run --analyze

# Run with result exports
uv run python -m examples.prompt_engineering.cli run \
  --csv-output results.csv \
  --json-output results.json \
  --html-output results.html \
  --analyze
```

### Using Custom Configuration
```bash
# Run with custom configuration file
uv run python -m examples.prompt_engineering.cli run \
  --config-path examples/prompt_engineering/config_test.json
```

## Customizing for Your Use Case

### 1. Define Your Prompt Variations
Edit `config.py` to add your own prompt templates:

```python
MY_CUSTOM_PROMPTS = [
    {
        "name": "custom-strategy-1",
        "template": "Your custom prompt template: {input}",
        "description": "Description of your first strategy",
        "metadata": {"strategy": "custom-1"}
    },
    {
        "name": "custom-strategy-2", 
        "template": "Another approach: {input}",
        "description": "Description of your second strategy",
        "metadata": {"strategy": "custom-2"}
    }
]
```

### 2. Configure Models
Add real models by updating the model configuration:

```json
{
  "models": [
    {
      "name": "gpt-4o",
      "provider": "litellm",
      "provider_options": {
        "api_key": "${OPENAI_API_KEY}"
      },
      "description": "OpenAI GPT-4o model"
    },
    {
      "name": "claude-3-opus",
      "provider": "litellm", 
      "provider_options": {
        "api_key": "${ANTHROPIC_API_KEY}"
      },
      "description": "Anthropic Claude 3 Opus model"
    }
  ]
}
```

### 3. Use Different Datasets
Support for various dataset types:
- Built-in demo datasets
- Hugging Face datasets (math500_hf)
- Local datasets (math500_local) 
- Inline custom datasets

### 4. Add Custom Metrics
You can extend the metric system to include domain-specific measures:

```python
from themis.interfaces import Metric
from themis.core.entities import MetricScore

class CustomMetric(Metric):
    def compute(self, prediction, references, metadata=None):
        # Your custom evaluation logic
        value = some_evaluation_function(prediction, references)
        return MetricScore(
            metric_name="CustomMetric",
            value=value,
            details={"info": "custom details"},
            metadata=metadata or {}
        )
```

## Analyzing Results

The example includes tools for analyzing results:

### Direct Analysis
The built-in analysis shows accuracy by prompt strategy:

```
Prompt Strategy Comparison:
--------------------------------------------------
zero-shot            | Accuracy: 0.250 (1/4)
few-shot             | Accuracy: 0.500 (2/4) 
chain-of-thought     | Accuracy: 0.750 (3/4)
```

### Export Analysis
Use the analysis script on exported CSV files:

```bash
python examples/prompt_engineering/results_analysis.py results.csv analysis_report.md
```

## Real-World Usage Patterns

### A/B Testing Prompts
Compare multiple prompt variations to identify the most effective approach for your use case.

### Model Prompt Optimization
Test how different prompting strategies work across various model architectures and providers.

### Benchmark Evaluation
Evaluate prompt performance on standard datasets like MATH, MMLU, or custom benchmarks.

### Systematic Experimentation
Run grid searches over prompt variations, models, and sampling parameters systematically.

## Project Structure

- `config.py`: Configuration models and defaults
- `experiment.py`: Core experiment implementation
- `prompts.py`: Prompt template definitions  
- `datasets.py`: Dataset loading utilities
- `cli.py`: Command-line interface
- `results_analysis.py`: Analysis utilities
- `config.sample.json`: Sample configuration file

This example provides a comprehensive template for conducting systematic prompt engineering experiments with Themis, enabling efficient testing and comparison of different prompting strategies at scale.