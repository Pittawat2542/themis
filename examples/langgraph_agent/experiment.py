"""LangGraph Agent experiment setup."""
from typing import List, Dict, Any
from themis.core import entities

from .runner import LangGraphAgentRunner

def run_experiment(
    dataset: List[Dict[str, Any]],
    model: entities.ModelSpec,
    sampling: entities.SamplingConfig,
    storage_dir: str,
    run_id: str,
    resume: bool,
):
    """Run the LangGraph agent experiment."""
    
    print(f"Running LangGraph agent experiment with {len(dataset)} samples...")
    
    # Create the LangGraph agent runner
    # Note: We pass a dummy provider since the agent doesn't use it
    from themis.generation.clients import FakeMathModelClient
    dummy_provider = FakeMathModelClient()
    
    runner = LangGraphAgentRunner(provider=dummy_provider)
    
    # Create generation tasks and run them
    generation_results = []
    for item in dataset:
        # Create prompt render
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
        
        # Execute with agent
        record = runner._execute_task(task)
        generation_results.append(record)
    
    # Create a simple report
    class SimpleReport:
        def __init__(self, results):
            self.generation_results = results
            self.evaluation_report = None
    
    return SimpleReport(generation_results)

__all__ = ["run_experiment"]
