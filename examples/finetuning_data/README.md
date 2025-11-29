# Fine-tuning Data Generation Example

This example demonstrates how to use Themis to generate synthetic training data for fine-tuning LLMs.

## Features Demonstrated

- **Data Generation**: Generate responses from a model on a dataset
- **Quality Filtering**: Filter out failed or incorrect responses
- **Format Conversion**: Export to JSONL format for fine-tuning
- **Metadata Tracking**: Include model, temperature, and sample information

## How it Works

1. **Generate Responses**: Run the model on your dataset to generate candidate responses
2. **Filter**: Remove failed generations and optionally filter by correctness
3. **Export**: Save filtered data in JSONL format ready for fine-tuning

## Running the Example

```bash
# Generate data with default settings (only correct answers)
uv run python -m examples.finetuning_data.cli run

# Include all responses (even incorrect ones)
uv run python -m examples.finetuning_data.cli run --only-correct false

# Custom output path
uv run python -m examples.finetuning_data.cli run --output my_training_data.jsonl
```

## Output Format

The generated JSONL file contains one record per line:

```json
{"prompt": "What is 5 + 3?", "completion": "{\"answer\": \"8\"}", "metadata": {"sample_id": "q1", "model": "fake-model", "temperature": 0.7}}
{"prompt": "What is 10 * 4?", "completion": "{\"answer\": \"40\"}", "metadata": {"sample_id": "q2", "model": "fake-model", "temperature": 0.7}}
```

## Use Cases

### 1. Synthetic Data Generation
Generate training data when you have:
- Programmatic ground truth (e.g., math, code execution)
- Access to a strong teacher model
- Need for diverse training examples

### 2. Distillation
Use a large model to generate training data for a smaller model:
```python
# Configure with a strong model
model = entities.ModelSpec(identifier="gpt-4", provider="openai")
```

### 3. Data Augmentation
Generate additional training examples with variations:
```python
# Use different temperatures for diversity
sampling = entities.SamplingConfig(temperature=0.9, top_p=0.95, max_tokens=512)
```

## Customization

### Custom Filtering Logic

Extend `FinetuningDataFilter` to add custom filtering:

```python
class CustomFilter(FinetuningDataFilter):
    def filter_records(self, records, evaluations=None):
        filtered = super().filter_records(records, evaluations)
        
        # Add custom filtering logic
        return [r for r in filtered if len(r["completion"]) > 10]
```

### Different Output Formats

Modify `export_jsonl` to support other formats:

```python
def export_csv(self, records, output_path):
    import csv
    with open(output_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'completion'])
        writer.writeheader()
        writer.writerows(records)
```

## Integration with Fine-tuning Platforms

### OpenAI Fine-tuning

```bash
# Use the generated JSONL file directly
openai api fine_tunes.create \
  -t finetuning_data.jsonl \
  -m gpt-3.5-turbo
```

### Hugging Face

```python
from datasets import load_dataset
dataset = load_dataset('json', data_files='finetuning_data.jsonl')
```

## Best Practices

1. **Quality Over Quantity**: Filter aggressively for high-quality examples
2. **Diversity**: Use varied temperatures and prompts
3. **Validation**: Always validate a sample of generated data manually
4. **Versioning**: Track which model and config generated each dataset
5. **Evaluation**: Keep a holdout set to evaluate the fine-tuned model

## Related Examples

- **[prompt_engineering](../prompt_engineering/)**: Test different prompts before generating data
- **[projects](../projects/)**: Organize multiple data generation experiments
