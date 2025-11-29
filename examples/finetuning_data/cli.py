"""CLI for fine-tuning data generation example."""
import cyclopts
from typing import Optional
from themis.core import entities
from .experiment import run_experiment

app = cyclopts.App(name="finetuning-data")

@app.command
def run(
    output: str = "finetuning_data.jsonl",
    only_correct: bool = True,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    format: str = "jsonl",  # jsonl or csv
):
    """Generate fine-tuning data from a dataset.
    
    Args:
        output: Output file path.
        only_correct: Only include correct answers.
        min_length: Minimum response length in characters.
        max_length: Maximum response length in characters.
        format: Output format (jsonl or csv).
    """
    
    # Define inline dataset
    dataset = [
        {
            "unique_id": "q1",
            "prompt": "What is 5 + 3?",
            "reference": "8",
            "metadata": {"subject": "arithmetic", "dataset_id": "q1"}
        },
        {
            "unique_id": "q2",
            "prompt": "What is 10 * 4?",
            "reference": "40",
            "metadata": {"subject": "arithmetic", "dataset_id": "q2"}
        },
        {
            "unique_id": "q3",
            "prompt": "What is 15 - 7?",
            "reference": "8",
            "metadata": {"subject": "arithmetic", "dataset_id": "q3"}
        }
    ]
    
    # Define model and sampling
    model = entities.ModelSpec(identifier="fake-model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=512)
    
    result = run_experiment(
        dataset=dataset,
        model=model,
        sampling=sampling,
        output_path=output,
        only_correct=only_correct,
        min_length=min_length,
        max_length=max_length,
        export_format=format
    )
    
    print(f"\nData generation completed!")
    print(f"Total samples generated: {result['total_generated']}")
    print(f"Filtered samples: {result['filtered_count']}")
    print(f"Output file: {result['output_path']}")
    print(f"\nYou can now use this file for fine-tuning!")

if __name__ == "__main__":
    app()
