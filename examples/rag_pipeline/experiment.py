from typing import List, Dict, Any
from themis.core import entities
from themis.generation.runner import GenerationRunner
from themis.generation.clients import FakeMathModelClient

from .knowledge_base import SimpleKnowledgeBase
from .retriever import RetrievalAugmentedRunner

def run_experiment(
    knowledge_base: SimpleKnowledgeBase,
    dataset: List[Dict[str, Any]],
    model: entities.ModelSpec,
    sampling: entities.SamplingConfig,
    storage_dir: str,
    run_id: str,
    resume: bool,
):
    """Run the RAG experiment."""
    
    print(f"Running RAG experiment with {len(dataset)} samples...")
    
    # Create provider
    provider = FakeMathModelClient()
    
    # Create custom RAG runner
    runner = RetrievalAugmentedRunner(
        provider=provider,
        knowledge_base=knowledge_base
    )
    
    # Create generation tasks
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
        
        # Generate with RAG
        record = runner._generate_single(task)
        generation_results.append(record)
    
    # Create a simple report
    class SimpleReport:
        def __init__(self, results):
            self.generation_results = results
            self.evaluation_report = None
    
    return SimpleReport(generation_results)
