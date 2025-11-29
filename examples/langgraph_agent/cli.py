"""CLI for LangGraph agent example."""
import cyclopts
from themis.core import entities
from .experiment import run_experiment

app = cyclopts.App(name="langgraph-agent")

@app.command
def run(
    limit: int = 5,
    storage_dir: str = ".cache/langgraph_example",
    run_id: str = "langgraph-demo",
    resume: bool = True,
):
    """Run the LangGraph agent example."""
    
    # Define inline dataset
    dataset = [
        {
            "unique_id": "q1",
            "prompt": "What is 5 + 3?",
            "reference": "8",
            "metadata": {"subject": "math"}
        },
        {
            "unique_id": "q2",
            "prompt": "What is 10 * 4?",
            "reference": "40",
            "metadata": {"subject": "math"}
        }
    ]
    
    # Define model and sampling (not actually used by agent, but required by Themis API)
    model = entities.ModelSpec(identifier="langgraph-agent", provider="custom")
    sampling = entities.SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=512)
    
    report = run_experiment(
        dataset=dataset,
        model=model,
        sampling=sampling,
        storage_dir=storage_dir,
        run_id=run_id,
        resume=resume
    )
    
    print("\nExperiment completed!")
    
    # Show an example of agent execution
    if hasattr(report, 'generation_results') and report.generation_results:
        record = report.generation_results[0]
        print("\n--- Example Agent Execution ---")
        print(f"Problem: {record.task.prompt.text}")
        if record.output:
            print(f"Answer: {record.output.text}")
            if record.output.raw:
                print(f"Agent Plan: {record.output.raw.get('plan', 'N/A')}")
                print(f"Agent Steps: {record.metrics.get('agent_steps', 'N/A')}")

if __name__ == "__main__":
    app()
