"""Custom dataset example using themis.evaluate()."""

import themis

DATASET = [
    {"id": "q1", "question": "What is 10 + 5?", "answer": "15"},
    {"id": "q2", "question": "What is 20 - 8?", "answer": "12"},
    {"id": "q3", "question": "What is 6 * 7?", "answer": "42"},
]

report = themis.evaluate(
    DATASET,
    model="fake:fake-math-llm",
    prompt="Q: {question}\nA:",
    metrics=["exact_match", "response_length"],
    temperature=0.0,
    max_tokens=128,
    workers=2,
    storage=".cache/experiments",
    resume=False,
)

print("Samples:", len(report.generation_results))
for name, aggregate in sorted(report.evaluation_report.metrics.items()):
    print(f"{name}: mean={aggregate.mean:.4f} (n={aggregate.count})")
