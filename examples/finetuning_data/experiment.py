"""Fine-tuning data generation experiment."""
from typing import List, Dict, Any, Optional
from themis.core import entities
from themis.generation.clients import FakeMathModelClient

from .pipeline import FinetuningDataFilter

def run_experiment(
    dataset: List[Dict[str, Any]],
    model: entities.ModelSpec,
    sampling: entities.SamplingConfig,
    output_path: str,
    only_correct: bool = True,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    export_format: str = "jsonl",
):
    """Generate fine-tuning data from a dataset.
    
    Args:
        dataset: List of dataset items.
        model: Model specification.
        sampling: Sampling configuration.
        output_path: Path to output file.
        only_correct: Only include correct responses.
        min_length: Minimum response length.
        max_length: Maximum response length.
        export_format: Output format (jsonl or csv).
    """
    
    print(f"Generating fine-tuning data from {len(dataset)} samples...")
    
    # Create provider
    provider = FakeMathModelClient()
    
    # Generate responses for all samples
    generation_results = []
    for item in dataset:
        #Create prompt render
        prompt_spec = entities.PromptSpec(name="default", template="{prompt}")
        prompt_render = entities.PromptRender(
            spec=prompt_spec,
            text=item["prompt"],
            context={"prompt": item["prompt"]}
        )
        
        # Create task
        task = entities.GenerationTask(
            prompt=prompt_render,
            model=model,
            sampling=sampling,
            metadata=item.get("metadata", {})
        )
        
        # Generate
        record = provider.generate(task)
        generation_results.append(record)
    
    # Filter and export
    filter = FinetuningDataFilter(
        only_correct=only_correct,
        min_length=min_length,
        max_length=max_length
    )
    filtered_records = filter.filter_records(generation_results)
    
    if export_format.lower() == "csv":
        filter.export_csv(filtered_records, output_path)
    else:
        filter.export_jsonl(filtered_records, output_path)
    
    return {
        "total_generated": len(generation_results),
        "filtered_count": len(filtered_records),
        "output_path": output_path,
        "format": export_format
    }

__all__ = ["run_experiment"]
