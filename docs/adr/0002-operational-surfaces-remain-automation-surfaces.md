---
diataxis: reference
audience: contributors
goal: Record how operational conveniences fit Themis without replacing executable Python definitions.
---

# Operational Surfaces Remain Automation Surfaces

Themis may add operational conveniences such as Benchmark Kits, Suite Definitions, Runtime Presets, Execution Resource Plans, Provider Telemetry, queue manifests, batch requests, and infrastructure adapters. These features exist to make evaluations easier to launch, inspect, compare, and operate.

They do not replace reviewed Python **Executable Definitions** as the canonical source of experiment meaning. A Benchmark Kit may help construct an `Experiment`; a Suite Definition may expand into several experiments; a Runtime Preset may resolve common configuration. The reviewed executable definition and its compiled `RunSnapshot` remain the authority for what was evaluated.

The rule is: runtime resource allocation is provenance unless it changes logical experiment behavior. Provider worker counts, stage concurrency, queue roots, batch roots, retry delays, telemetry sinks, and scheduler hints explain how a run happened. They do not redefine the logical experiment or determine run identity by themselves.

Operational features must preserve Themis-owned replaceable boundaries. Provider integrations, code execution, storage, reporting, and launch machinery should sit behind Themis interfaces rather than importing another evaluation framework or coupling core behavior to one infrastructure provider.

This ADR extends ADR-0001: config files, CLI arguments, submission manifests, worker queues, batch requests, suites, and presets are **Automation Surfaces**. They may transport, launch, override, or group executable definitions, but they are not the canonical source of defensible research meaning.
