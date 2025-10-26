# Experiment Data Flow

```mermaid
flowchart TD
    CLI["CLI (Cyclopts)"] --> Config[Config Loader / Overrides]
    Config --> Project["Project (Experiment Collection)"]
    Project --> Builder[ExperimentBuilder]
    Config --> Builder
    Patterns["Experiment Patterns (x-ablation)"] --> Project
    Builder --> Plan[GenerationPlan]
    Builder --> GStrategyResolver[Generation Strategy Resolver]
    Builder --> EStrategyResolver[Evaluation Strategy Resolver]
    Plan --> GStrategy[Generation Strategy]
    GStrategyResolver --> GStrategy
    GStrategy --> Runner[GenerationRunner]
    Runner -->|max_parallel| Providers
    Runner --> Router[Provider Router]
    Router --> Providers[Model Providers]
    Providers --> Attempts[Generation Attempts]
    Attempts --> Aggregated[Aggregated Generation Records]
    Aggregated --> Storage[Storage / Resume Cache]
    Aggregated --> EvalPipeline[Evaluation Pipeline]
    EvalPipeline --> EStrategy[Evaluation Strategy]
    EStrategyResolver --> EStrategy
    EStrategy --> Metrics[Metric Aggregates]
    Metrics --> Report[Experiment Report]
    Storage --> Report
    Report --> Output[CLI Output]
    Report --> Exports["Report Export (CSV / HTML / JSON)"]
    Patterns --> Charts["Charts / Ablation Data"]
    Charts --> Exports
```

Projects now sit between configuration and experiment assembly so multiple experiments (or pattern-generated variants) can share metadata and storage settings. Pattern helpers such as `XAblationPattern` feed both project registration and chart payloads that render in the HTML export. The `Report Export` node illustrates the bundled CSV/HTML/JSON outputs that CLIs can emit after any run.
