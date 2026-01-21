# Themis Examples Cookbook

This directory contains practical, hands-on examples that teach you how to use Themis effectively. Each example builds on the previous ones, forming a comprehensive learning path.

## Learning Path

### 🚀 [getting_started](getting_started/) - Your First Experiment

**Start here if you're new to Themis.**

Learn the absolute basics:
- Define a prompt template
- Configure models and sampling
- Run an experiment programmatically or via CLI
- View and export results

**Time to complete:** 15 minutes

**Key concepts:** Prompt templates, model specs, sampling configs, exact match metrics

**Example commands:**
```bash
# Run your first experiment
uv run python -m examples.getting_started.cli run

# Preview configuration
uv run python -m examples.getting_started.cli run --dry-run

# Export results
uv run python -m examples.getting_started.cli run --csv-output results.csv
```

---

### ⚙️ [config_file](config_file/) - Configuration-Driven Workflows

**Perfect for systematic experimentation.**

Learn how to drive experiments entirely through configuration files:
- Structure config files for reproducibility
- Compare multiple models and sampling strategies
- Override configs from the command line
- Use resumability and caching effectively

**Time to complete:** 20 minutes

**Key concepts:** JSON configs, CLI overrides, grid searches, resumability

**Example commands:**
```bash
# Run with a config file
uv run python -m examples.config_file.cli run --config-path compare_sampling.json

# Override from CLI
uv run python -m examples.config_file.cli run --config-path config.sample.json --run-id my-run

# Grid search: 2 models × 4 temperatures
uv run python -m examples.config_file.cli run --config-path grid_search.json
```

---

### 📝 [prompt_engineering](prompt_engineering/) - Systematic Prompt Comparison

**Master prompt engineering techniques.**

Learn how to systematically test and compare different prompting strategies:
- Zero-shot vs few-shot vs chain-of-thought prompting
- Compare prompt effectiveness across multiple models
- Use built-in metrics for evaluation
- Export results for analysis and reporting

**Time to complete:** 25 minutes

**Key concepts:** Prompt variations, strategy comparison, effectiveness analysis

**Example commands:**
```bash
# Run prompt engineering experiment
uv run python -m examples.prompt_engineering.cli run

# Run with analysis
uv run python -m examples.prompt_engineering.cli run --analyze

# Export results for analysis
uv run python -m examples.prompt_engineering.cli run \
  --csv-output results.csv \
  --html-output results.html
```

---

### 📊 [projects](projects/) - Organizing Multiple Experiments

**Scale up to research-level organization.**

Learn how to manage complex projects with multiple experiments:
- Group related experiments in a Project
- Share configurations across experiments
- Run experiments selectively
- Compare results across different approaches
- Maintain reproducible research workflows

**Time to complete:** 45 minutes

**Key concepts:** Project structure, experiment definitions, metadata, tags, systematic evaluation

**Example commands:**
```bash
# List all experiments in project
uv run python -m examples.projects.cli list-experiments

# Run specific experiment
uv run python -m examples.projects.cli run --experiment zero-shot

# Run with custom settings
uv run python -m examples.projects.cli run \
  --experiment few-shot \
  --storage-dir .cache/my-project \
  --limit 10
```

---

### 🔧 [advanced](advanced/) - Advanced Customization

**For power users who need full control.**

Learn how to customize and extend Themis:
- **Storage optimization** with StorageConfig (60-75% space savings)
- **Multi-value references** for complex evaluation tasks
- **Custom reference selectors** that take precedence
- Override generation loops with custom runners
- Create custom evaluation pipelines
- Implement domain-specific metrics
- Add instrumentation and debugging
- Build agentic workflows with multi-step generation

**Time to complete:** 60 minutes

**Key concepts:** StorageConfig, multi-value references, custom runners, evaluation pipelines, metrics, agentic workflows

**Example commands:**
```bash
# Run with subject-aware evaluation
uv run python -m examples.advanced.cli run --enable-subject-breakdown

# Use chain-of-thought prompting
uv run python -m examples.advanced.cli run --prompt-style cot

# Full custom configuration
uv run python -m examples.advanced.cli run \
  --prompt-style cot \
  --enable-subject-breakdown \
  --config-path examples/advanced/config.sample.json
```

---

### 🔍 [rag_pipeline](rag_pipeline/) - Retrieval-Augmented Generation

**Demonstrate RAG patterns with Themis.**

Learn how to implement retrieval-augmented generation:
- Custom generation runner with retrieval
- In-memory vector store using numpy
- Prompt augmentation with retrieved context
- Task metadata tracking for analysis

**Time to complete:** 30 minutes

**Key concepts:** Custom runners, retrieval, prompt augmentation, RAG workflows

**Example commands:**
```bash
# Run RAG example
uv run python -m examples.rag_pipeline.cli run
```

---

### 🤖 [langgraph_agent](langgraph_agent/) - Agentic Workflows

**Integrate LangGraph agents with Themis evaluation.**

Learn how to wrap LangGraph agents for systematic testing:
- LangGraph state machine integration
- Multi-step reasoning (planning → execution)
- Agent metadata and tracking
- Systematic evaluation of agent outputs

**Time to complete:** 35 minutes

**Key concepts:** Agentic workflows, state machines, multi-step reasoning, LangGraph integration

**Dependencies:** Requires `langgraph`
```bash
pip install langgraph
```

**Example commands:**
```bash
# Run LangGraph agent example
uv run python -m examples.langgraph_agent.cli run
```

---

### 📊 [finetuning_data](finetuning_data/) - Synthetic Training Data

**Generate high-quality fine-tuning datasets.**

Learn how to create synthetic training data:
- Generate model responses on datasets
- Filter for quality and correctness
- Export to JSONL for fine-tuning
- Track provenance with metadata

**Time to complete:** 25 minutes

**Key concepts:** Data generation, quality filtering, JSONL export, fine-tuning preparation

**Example commands:**
```bash
# Generate fine-tuning data
uv run python -m examples.finetuning_data.cli run

# Include all responses (not just correct ones)
uv run python -m examples.finetuning_data.cli run --only-correct false
```

---

## Additional Examples

### 🔌 [litellm_example](litellm_example/) - LiteLLM Provider Integration

**Connect to 100+ LLM providers.**

Learn how to use LiteLLM to connect to various LLM providers:
- OpenAI, Anthropic, Azure, AWS Bedrock, Google AI
- Local LLMs (Ollama, LM Studio, vLLM)
- Provider configuration and authentication
- Timeout, retries, and parallelism settings

See the [litellm_example README](litellm_example/README.md) for detailed setup instructions.

---

### ⚖️ [judge_evaluation](judge_evaluation/) - LLM-as-a-Judge

**Evaluate outputs using LLM judges.**

Learn how to use LLM-based evaluation:
- RubricJudgeMetric for criteria-based scoring
- ConsistencyMetric for inter-judge agreement
- Multiple judge evaluation
- Judge-based quality assessment

See the [judge_evaluation README](judge_evaluation/README.md) for more details.

---

## Quick Reference

### Common Tasks

**Run a quick test:**
```bash
cd examples/getting_started
uv run python -m examples.getting_started.cli run --dry-run
```

**Compare models:**
```bash
cd examples/config_file
uv run python -m examples.config_file.cli run --config-path compare_models.json
```

**Evaluate with prompt engineering:**
```bash
cd examples/prompt_engineering
uv run python -m examples.prompt_engineering.cli run --analyze
```

**Organize experiments:**
```bash
cd examples/projects
uv run python -m examples.projects.cli list-experiments
uv run python -m examples.projects.cli run --experiment zero-shot
```

**Custom behavior:**
```bash
cd examples/advanced
uv run python -m examples.advanced.cli run --enable-subject-breakdown
```

**RAG pipeline:**
```bash
cd examples/rag_pipeline
uv run python -m examples.rag_pipeline.cli run
```

**LangGraph agent:**
```bash
cd examples/langgraph_agent
uv run python -m examples.langgraph_agent.cli run
```

**Generate fine-tuning data:**
```bash
cd examples/finetuning_data
uv run python -m examples.finetuning_data.cli run
```

### Configuration Patterns

**Single model, single sampling:**
```json
{
  "models": [{"name": "model-1", "provider": "fake"}],
  "samplings": [{"name": "greedy", "temperature": 0.0, "max_tokens": 512}]
}
```

**Grid search (M models × N samplings):**
```json
{
  "models": [
    {"name": "model-1", "provider": "fake"},
    {"name": "model-2", "provider": "fake"}
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512},
    {"name": "creative", "temperature": 0.8, "max_tokens": 512}
  ]
}
```

**OpenAI-compatible endpoint:**
```json
{
  "models": [{
    "name": "my-model",
    "provider": "openai-compatible",
    "provider_options": {
      "base_url": "http://localhost:1234/v1",
      "api_key": "not-needed",
      "model_mapping": {"my-model": "actual-model-name"}
    }
  }]
}
```

## Troubleshooting

### Common Issues

**Can't find the example:**
```bash
# Always run from project root
cd /path/to/themis
uv run python -m examples.getting_started.cli run
```

**Connection refused (OpenAI compatible):**
1. Verify server is running: `curl http://localhost:1234/v1/models`
2. Check port number in config
3. Try `127.0.0.1` instead of `localhost`

**Results not updating:**
- Check `resume: true` in config - may be using cached results
- Try `--resume false` or use a new `run_id`

**Slow performance:**
- Reduce `n_parallel` if server is overloaded
- Use `limit` to test with fewer samples first
- Check CPU/GPU usage on server

## File Structure

```
examples/
├── README.md                      # This file
├── getting_started/               # Basics
│   ├── README.md
│   ├── cli.py
│   ├── config.py
│   ├── config.sample.json
│   ├── datasets.py
│   └── experiment.py
├── config_file/                   # Configuration-driven
│   ├── README.md
│   ├── cli.py
│   ├── config.sample.json
│   ├── compare_sampling.json
│   ├── compare_models.json
│   └── grid_search.json
├── prompt_engineering/            # Prompt strategy comparison
│   ├── README.md
│   ├── USAGE.md
│   ├── cli.py
│   ├── config.py
│   ├── config.sample.json
│   ├── prompts.py
│   ├── datasets.py
│   ├── experiment.py
│   └── results_analysis.py
├── projects/                      # Multi-experiment projects
│   ├── README.md
│   ├── cli.py
│   └── project_setup.py
└── advanced/                      # Advanced customization
    ├── README.md
    ├── cli.py
    ├── generation.py
    ├── pipeline.py
    └── config.sample.json
├── rag_pipeline/                  # RAG implementation
│   ├── README.md
│   ├── cli.py
│   ├── knowledge_base.py
│   ├── retriever.py
│   ├── experiment.py
│   └── config.py
├── langgraph_agent/               # Agentic workflows
│   ├── README.md
│   ├── cli.py
│   ├── agent.py
│   ├── runner.py
│   └── experiment.py
├── finetuning_data/               # Synthetic data generation
│   ├── README.md
│   ├── cli.py
│   ├── pipeline.py
│   └── experiment.py
├── litellm_example/               # LiteLLM provider integration
│   └── ...
└── judge_evaluation/              # LLM-as-a-judge evaluation
    └── ...
```

## Tips for Learning

1. **Start with getting_started** - Don't skip ahead! The basics are essential.

2. **Run every example** - Reading isn't enough. Run the code to understand it.

3. **Modify configs** - Copy `config.sample.json` and experiment with different values.

4. **Use dry-run mode** - Preview what will happen: `--dry-run`

5. **Start small** - Use `limit: 5` or `--n-records 5` for quick iteration.

6. **Check storage dirs** - Results are cached in `storage_dir` for resumability.

7. **Export results** - Use `--csv-output` and `--html-output` for analysis.

8. **Read error messages** - They usually tell you exactly what's wrong.

## New Features Quick Reference

### Storage Optimization (See: advanced example)

Optimize storage for large experiments:

```python
from themis.experiment.storage import StorageConfig

# Production: 60-75% storage reduction
config = StorageConfig(
    save_raw_responses=False,    # Saves ~5MB per 1.5K samples
    compression="gzip",           # 50-60% reduction
    deduplicate_templates=True,   # Saves ~627KB per 1.5K samples
)

# Result: 18.5MB → 3-5MB for 1,500 samples
```

### Quick Results Viewing

View experiment results without parsing large files:

```bash
# View summary (1KB file vs 1.6MB report)
uv run python -m themis.cli results-summary --run-id run-20260118-032014

# List all runs
uv run python -m themis.cli results-list

# List 10 most recent
uv run python -m themis.cli results-list --limit 10
```

Export summaries in code:

```python
from themis.experiment.export import export_summary_json

export_summary_json(report, "outputs/run-123/summary.json", run_id="run-123")
```

### Multi-Value References (See: advanced example)

Use dict values for complex evaluation:

```python
from themis.core.entities import Reference

# Multi-value reference
ref = Reference(
    kind="task",
    value={"answer": 42, "steps": [...], "constraints": [...]}
)

# In metric
def compute(self, *, prediction, references, metadata=None):
    ref = references[0]
    if isinstance(ref, dict):
        answer = ref["answer"]
        steps = ref["steps"]
```

### Custom Reference Selectors (See: advanced example)

Extract custom references that take precedence:

```python
from themis.evaluation import EvaluationPipeline

def my_selector(record):
    return {
        "answer": record.task.reference.value,
        "extra": record.task.metadata.get("extra_data")
    }

pipeline = EvaluationPipeline(
    extractor=extractor,
    metrics=[metric],
    reference_selector=my_selector  # Takes precedence
)
```

### Clear Extractor Contract

Metrics receive **extracted** output (not raw text):

```python
from themis.interfaces import Metric

class MyMetric(Metric):
    def compute(self, *, prediction, references, metadata=None):
        # ✅ prediction is already extracted
        # DON'T try to extract again!
        is_correct = prediction == references[0]
        return MetricScore(metric_name=self.name, value=1.0 if is_correct else 0.0)
```

## Next Steps

After completing these examples:

1. **Read the docs**: Check `docs/ADDING_COMPONENTS.md` for extension points
2. **Review the source**: `themis/` contains well-documented code
3. **Check the tests**: `tests/` show usage patterns
4. **Build your own**: Create a new experiment for your use case

## Getting Help

- **Documentation**: See `docs/` directory
- **API Reference**: Check docstrings in `themis/` modules
- **Issues**: Search or create issues on the repository
- **Examples**: These examples are your best resource!

## Contributing Examples

Have a useful example to share? Contributions are welcome!

1. Follow the existing structure (README, CLI, configs)
2. Include clear explanations and multiple use cases
3. Test all commands in the README
4. Submit a pull request

Happy experimenting! 🚀