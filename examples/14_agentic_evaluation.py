"""Example: Agentic evaluation using StatefulTaskExecutor.

This script demonstrates how to evaluate a multi-turn agent that
uses a `StatefulTaskExecutor` to solve problems through intermediate steps.
By registering it as a provider, you can use `themis.evaluate()` normally.
"""

import themis
from themis.interfaces import StatefulTaskExecutor
from themis.core.entities import GenerationTask, GenerationRecord, ModelOutput


class SimpleMathAgent(StatefulTaskExecutor):
    """A mock agent that takes two steps to solve a math problem."""

    def __init__(self, **kwargs):
        # provider instances often take kwargs for initialization
        self.kwargs = kwargs

    def step(self, prompt, **kwargs) -> ModelOutput:
        # Our mock step doesn't do much, it's just for the interface
        return ModelOutput(text="Thought...")

    def execute_task(self, task: GenerationTask) -> GenerationRecord:
        history = [task.prompt.text]

        # Step 1: Think
        thought_output = self.step("Think about it")
        history.append(thought_output.text)

        # Step 2: Extract real answer
        prompt_text = task.prompt.text
        words = prompt_text.split()
        if "2" in words:
            final_answer = "4"
        elif "life" in words:
            final_answer = "42"
        else:
            final_answer = "0"

        history.append(f"Answer: {final_answer}")

        output = ModelOutput(text=final_answer, raw={"history": history})

        return GenerationRecord(task=task, output=output, error=None)


# Register our agent as a provider
themis.register_provider("simple_agent", lambda **kwargs: SimpleMathAgent(**kwargs))


def main():
    dataset = [
        {"id": "1", "question": "What is 2 + 2?", "answer": "4"},
        {"id": "2", "question": "What is the meaning of life?", "answer": "42"},
        {"id": "3", "question": "What is 10 + 10?", "answer": "20"},
    ]

    print("Running evaluate() with our stateful agent...")
    report = themis.evaluate(
        dataset,
        model="simple_agent:v1",  # "provider:model"
        prompt="Solve: {question}",
        reference_field="answer",
        run_id="agentic_eval_demo",
    )

    print(f"\nOverall Exact Match: {report.metric('exact_match').mean:.2%}")

    # Inspect the history of the first record
    first_record = report.generation_results[0]
    print("\n--- Example Agent History ---")
    history_items = (
        first_record.output.raw.get("history", [])
        if first_record.output and isinstance(first_record.output.raw, dict)
        else []
    )
    for idx, item in enumerate(history_items):
        print(f"[{idx}] {item}")


if __name__ == "__main__":
    main()
