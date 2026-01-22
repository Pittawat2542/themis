"""Distributed execution example - Scale with Ray.

This example shows how to use distributed execution for faster evaluations.
Note: This requires Ray to be installed and will be fully implemented in Phase 3.
"""

import themis

# Run evaluation with distributed execution
# (This will fail for now since distributed execution is not yet implemented)
print("Distributed execution example")
print("Note: This feature will be implemented in Phase 3")
print()

# For now, run with local execution
report = themis.evaluate(
    "demo",
    model="fake-math-llm",
    limit=3,
    # distributed=True,  # Uncomment when Phase 3 is complete
    # workers=8,         # Number of parallel workers
    # storage="s3://my-bucket/runs",  # Cloud storage for distributed runs
)

print(f"Completed {len(report.generation_results)} samples")
print("\nTo enable distributed execution (Phase 3):")
print("  1. Install Ray: pip install ray[default]")
print("  2. Set distributed=True")
print("  3. Configure cloud storage (S3, GCS, or Azure)")
