"""Custom dataset example - Evaluate on your own data.

This example shows how to run evaluations on custom datasets
with your own prompts and metrics.
"""

import themis

# Define custom dataset
my_dataset = [
    {"id": "q1", "question": "What is 10 + 5?", "answer": "15"},
    {"id": "q2", "question": "What is 20 - 8?", "answer": "12"},
    {"id": "q3", "question": "What is 6 * 7?", "answer": "42"},
]

# Run evaluation with custom prompt
report = themis.evaluate(
    dataset=my_dataset,
    model="fake-math-llm",
    prompt="Q: {question}\nA:",
    metrics=["exact_match"],
)

# Print results
print("\nCustom Dataset Evaluation:")
print(f"Total samples: {len(report.generation_results)}")
print(f"Successful: {len([r for r in report.generation_results if r.error is None])}")

if report.evaluation_report.aggregates:
    for aggregate in report.evaluation_report.aggregates:
        print(f"{aggregate.metric_name}: {aggregate.mean:.2%}")

# Show individual results
print("\nIndividual Results:")
for record in report.generation_results:
    sample_id = record.task.metadata.get("dataset_id")
    output_text = record.output.text if record.output else "ERROR"
    print(f"  {sample_id}: {output_text}")
