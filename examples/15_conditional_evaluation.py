"""Conditional evaluation example showing AdaptiveEvaluationPipeline.

This example demonstrates how to apply different evaluation metrics to different
samples based on their metadata (e.g., difficulty).
"""

from themis.core.entities import (
    GenerationRecord,
    GenerationTask,
    ModelOutput,
    ModelSpec,
    PromptRender,
    PromptSpec,
    Reference,
    SamplingConfig,
)
from themis.evaluation.conditional import (
    AdaptiveEvaluationPipeline,
    ConditionalMetric,
    select_by_difficulty,
)
from themis.evaluation.extractors import IdentityExtractor
from themis.evaluation.metrics.exact_match import ExactMatch
from themis.evaluation.metrics.response_length import ResponseLength

print("Building test records...")

# Create tasks with varying difficulties
tasks = [
    GenerationTask(
        prompt=PromptRender(
            spec=PromptSpec("test", "{q}"),
            text="What is 2+2?",
        ),
        model=ModelSpec("fake-math-llm", "fake"),
        sampling=SamplingConfig(),
        reference=Reference(kind="answer", value="4"),
        metadata={"difficulty": "easy"},
    ),
    GenerationTask(
        prompt=PromptRender(
            spec=PromptSpec("test", "{q}"),
            text="Write a long essay on the Roman Empire.",
        ),
        model=ModelSpec("fake-math-llm", "fake"),
        sampling=SamplingConfig(),
        reference=Reference(kind="answer", value="Rome was an empire."),
        metadata={"difficulty": "hard"},
    ),
]

# Simulate generation results
records = [
    GenerationRecord(
        task=tasks[0],
        output=ModelOutput("4"),
        error=None,
    ),
    GenerationRecord(
        task=tasks[1],
        output=ModelOutput("Rome was an empire that lasted a long time..."),
        error=None,
    ),
]

print("\n--- Example 1: AdaptiveEvaluationPipeline ---")

# Create an adaptive pipeline that chooses metrics based on difficulty
adaptive_pipeline = AdaptiveEvaluationPipeline(
    extractor=IdentityExtractor(),
    metric_selector=select_by_difficulty(
        easy_metrics=[ExactMatch()],
        medium_metrics=[ExactMatch(), ResponseLength()],
        # exact match is too brittle for essays, only check length
        hard_metrics=[ResponseLength()],
        difficulty_field="difficulty",
    ),
)

# Run evaluation directly using the pipeline
report = adaptive_pipeline.evaluate(records)

for eval_record, gen_record in zip(report.records, records):
    difficulty = gen_record.task.metadata.get("difficulty")
    print(f"\nSample (Difficulty: {difficulty}):")
    print(f"Output: {gen_record.output.text}")
    print("Metrics applied:")
    for score in eval_record.scores:
        print(f"  - {score.metric_name}: {score.value:.2f}")


print("\n--- Example 2: ConditionalMetric ---")

# Alternatively, wrap a single metric to only run under specific conditions
# (Here we only run ExactMatch if it's NOT a hard question)
conditional_em = ConditionalMetric(
    metric=ExactMatch(),
    condition=lambda r: r.task.metadata.get("difficulty") != "hard",
)

for record in records:
    difficulty = record.task.metadata.get("difficulty")
    print(f"\nSample (Difficulty: {difficulty}):")

    if conditional_em.should_evaluate(record):
        # We manually use the extractor and metric interface to show invocation
        prediction = IdentityExtractor().extract(record.output.text)
        score = conditional_em.compute(
            prediction=prediction,
            references=[record.task.reference.value],
            metadata=record.task.metadata,
        )
        print(f"  Condition met! Score: {score.value:.2f}")
    else:
        # compute_or_default handles skipping when condition fails
        score = conditional_em.compute_or_default(
            record,
            prediction=IdentityExtractor().extract(record.output.text),
            references=[record.task.reference.value],
        )
        print("  Condition NOT met! Metric skipped.")
        print(f"  Default score returned: {score}")

print("\nConditional evaluation complete!")
