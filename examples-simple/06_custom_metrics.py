"""Example: Custom Metric Registration

This example demonstrates how to register and use custom metrics in Themis.

Custom metrics allow you to define your own evaluation criteria beyond the
built-in metrics like ExactMatch and MathVerify.
"""

from dataclasses import dataclass

import themis
from themis.core.entities import MetricScore
from themis.interfaces import Metric


@dataclass
class WordCountMetric(Metric):
    """A simple metric that checks if the response has enough words."""
    
    min_words: int = 10
    
    def __post_init__(self):
        self.name = f"word_count_min_{self.min_words}"
    
    def compute(self, *, prediction, references=None, metadata=None):
        """Check if prediction has at least min_words words."""
        word_count = len(str(prediction).split())
        meets_requirement = word_count >= self.min_words
        
        return MetricScore(
            metric_name=self.name,
            value=1.0 if meets_requirement else 0.0,
            details={
                "word_count": word_count,
                "min_required": self.min_words,
            },
            metadata=metadata or {},
        )


@dataclass
class ContainsKeywordMetric(Metric):
    """A metric that checks if the response contains specific keywords."""
    
    keyword: str = "because"
    
    def __post_init__(self):
        self.name = f"contains_{self.keyword}"
    
    def compute(self, *, prediction, references=None, metadata=None):
        """Check if prediction contains the keyword."""
        text = str(prediction).lower()
        contains = self.keyword.lower() in text
        
        return MetricScore(
            metric_name=self.name,
            value=1.0 if contains else 0.0,
            details={
                "keyword": self.keyword,
                "found": contains,
            },
            metadata=metadata or {},
        )


def main():
    """Demonstrate custom metric registration."""
    print("=" * 60)
    print("Custom Metric Registration Example")
    print("=" * 60)
    
    # Step 1: Register custom metrics
    print("\n1. Registering custom metrics...")
    themis.register_metric("word_count", WordCountMetric)
    themis.register_metric("contains_keyword", ContainsKeywordMetric)
    print("   ✓ Registered 'word_count' metric")
    print("   ✓ Registered 'contains_keyword' metric")
    
    # Step 2: Check registered metrics
    print("\n2. Checking registered metrics...")
    registered = themis.get_registered_metrics()
    print(f"   Total custom metrics: {len(registered)}")
    for name, cls in registered.items():
        print(f"   - {name}: {cls.__name__}")
    
    # Step 3: Create a simple dataset
    print("\n3. Creating test dataset...")
    dataset = [
        {
            "id": "1",
            "question": "Why is the sky blue?",
            "answer": "The sky appears blue because of Rayleigh scattering.",
        },
        {
            "id": "2",
            "question": "What is 2+2?",
            "answer": "4",
        },
        {
            "id": "3",
            "question": "Explain photosynthesis.",
            "answer": "Photosynthesis is the process where plants convert light into energy.",
        },
    ]
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Step 4: Run evaluation with custom metrics
    print("\n4. Running evaluation with custom metrics...")
    print("   (Using fake model for demonstration)")
    
    report = themis.evaluate(
        dataset,
        model="fake",
        prompt="Q: {question}\nA:",
        metrics=["exact_match", "word_count", "contains_keyword"],
        limit=3,
    )
    
    # Step 5: Display results
    print("\n5. Results:")
    print("   " + "=" * 50)
    
    if report.evaluation_report.metrics:
        for metric_name, aggregate in report.evaluation_report.metrics.items():
            print(f"   {metric_name}: {aggregate.mean:.2%} (avg over {aggregate.count} samples)")
    
    print("\n6. Interpretation:")
    print("   - word_count_min_10: Measures if responses have at least 10 words")
    print("   - contains_because: Checks if responses contain 'because'")
    print("   - ExactMatch: Built-in metric for exact string matching")
    print("\n   The fake model generates JSON responses, which have >10 words")
    print("   but don't contain 'because', and don't match the expected answers.")
    
    print("\n" + "=" * 60)
    print("✓ Custom metrics working successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Define your own metric class")
    print("  2. Register it with themis.register_metric()")
    print("  3. Use it in evaluate(metrics=[...])")
    print("=" * 60)


if __name__ == "__main__":
    main()
