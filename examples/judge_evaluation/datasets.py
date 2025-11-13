"""Dataset loading for judge evaluation example."""

from __future__ import annotations


DEMO_SAMPLES = [
    {
        "unique_id": "judge-demo-1",
        "problem": "What is 2 + 2?",
        "answer": "4",
        "candidate_solution": "The sum of 2 and 2 equals 4. This is a basic arithmetic operation.",
        "subject": "arithmetic",
    },
    {
        "unique_id": "judge-demo-2",
        "problem": "Convert (0,3) to polar coordinates.",
        "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
        "candidate_solution": "The point (0,3) in polar form is (3, π/2) because r = sqrt(0² + 3²) = 3 and θ = arctan(3/0) = π/2.",
        "subject": "precalculus",
    },
    {
        "unique_id": "judge-demo-3",
        "problem": "What is the square root of 16?",
        "answer": "4",
        "candidate_solution": "4, because 4 * 4 = 16",
        "subject": "arithmetic",
    },
]


def load_demo_dataset() -> list[dict[str, object]]:
    """Load demo dataset for judge evaluation."""
    return [dict(sample) for sample in DEMO_SAMPLES]
