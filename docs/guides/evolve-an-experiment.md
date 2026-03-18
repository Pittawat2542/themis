# Evolve a Benchmark

The benchmark-first model still supports incremental evolution against the same
storage root.

Typical changes:

- add a model
- add a prompt variant
- add a parse pipeline
- add a score overlay
- narrow or widen a dataset query

Worked example: `examples/09_experiment_evolution.py`

The baseline run uses one model and one prompt. The expanded run adds another
model and another prompt variant while reusing the same project storage root.
