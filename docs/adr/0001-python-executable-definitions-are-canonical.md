---
diataxis: reference
audience: contributors
goal: Record why reviewed Python executable definitions remain the canonical source of experiment meaning.
---

# Python Executable Definitions Are Canonical

Themis treats reviewed importable Python modules that construct `Experiment` objects as the canonical source of serious experiment meaning. Config files, CLI arguments, submission manifests, worker queues, and batch requests are automation surfaces: they may transport, launch, or override an executable definition, but they should not be documented as the canonical source of truth for defensible research. This keeps custom behavior reviewable in typed Python while still supporting shell-friendly and deferred execution workflows.
