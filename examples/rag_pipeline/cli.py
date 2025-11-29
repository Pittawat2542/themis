"""CLI for RAG example."""
import cyclopts
from themis.core import entities
from .experiment import run_experiment
from .knowledge_base import SimpleKnowledgeBase

app = cyclopts.App(name="rag-pipeline")

@app.command
def run(
    limit: int = 5,
    storage_dir: str = ".cache/rag_example",
    run_id: str = "rag-demo",
    resume: bool = True,
):
    """Run the RAG pipeline example."""
    
    # Define some knowledge base documents about Themis
    kb_docs = [
        "Themis is a lightweight experimentation harness for text-generation systems.",
        "Themis orchestrates prompt templates, LLM providers, generation strategies, evaluation metrics, and storage.",
        "Themis works with 100+ LLM providers via LiteLLM including OpenAI, Anthropic, Azure, AWS Bedrock, and Google AI.",
        "Themis has built-in evaluation metrics for exact match, math verification, and custom metrics.",
        "Themis supports systematic grid search over models, prompts, and sampling strategies.",
    ]
    
    # Initialize Knowledge Base
    print(f"Initializing knowledge base with {len(kb_docs)} documents...")
    kb = SimpleKnowledgeBase()
    kb.add_documents(kb_docs)
    
    # Define inline dataset
    dataset = [
        {
            "unique_id": "q1",
            "prompt": "What is Themis?",
            "reference": "Themis is a lightweight experimentation harness for text-generation systems.",
            "metadata": {"subject": "themis"}
        },
        {
            "unique_id": "q2",
            "prompt": "What providers does it support?",
            "reference": "It supports 100+ providers via LiteLLM including OpenAI, Anthropic, and local LLMs.",
            "metadata": {"subject": "themis"}
        }
    ]
    
    # Define model and sampling
    model = entities.ModelSpec(identifier="fake-model", provider="fake")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=512)
    
    report = run_experiment(
        knowledge_base=kb,
        dataset=dataset,
        model=model,
        sampling=sampling,
        storage_dir=storage_dir,
        run_id=run_id,
        resume=resume
    )
    
    print("\nExperiment completed!")
    if hasattr(report, 'evaluation_report') and hasattr(report.evaluation_report, 'metrics'):
        print(f"Metrics: {report.evaluation_report.metrics}")
    
    # Show an example of retrieval augmentation
    if hasattr(report, 'generation_results') and report.generation_results:
        record = report.generation_results[0]
        print("\n--- Example Generation ---")
        if 'original_prompt' in record.task.metadata:
            print(f"Original Prompt: {record.task.metadata['original_prompt']}")
        if 'retrieved_docs' in record.task.metadata:
            print(f"Retrieved Context: {record.task.metadata['retrieved_docs']}")
        print(f"Augmented Prompt:\n{record.task.prompt.text if hasattr(record.task.prompt, 'text') else record.task.prompt}")
        if record.output:
            print(f"Response: {record.output.text}")


if __name__ == "__main__":
    app()
