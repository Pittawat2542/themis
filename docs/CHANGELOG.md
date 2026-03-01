# Changelog

Canonical release history lives in the repository root at [`CHANGELOG.md`](https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md).

## Current Release

- Version: `1.4.0`
- [Release Notes 1.4.0](releases/1.4.0.md)

## Previous Releases
- Version: `1.3.0`
- Date: `2026-02-27`
- Highlights:
  - Comprehensive custom exception hierarchy.
  - Massively improved import times (~0.036s) by leveraging lazy loading.
  - Finalized removal of all legacy `ModelProvider` logic and deprecated items.

- Version: `1.2.1`
- Date: `2026-02-22`
- Highlights:
  - Refactored `ModelProvider` out in favor of `StatelessTaskExecutor` and `StatefulTaskExecutor`.
  - Added Advanced Orchestration guide and philosophy document.

- Version: `1.2.0`
- Date: `2026-02-20`
- Highlights:
  - Massive architectural cleanup removing defunct internal classes (`MetricPipeline`, `FlexibleGenerationPlan`).
  - Extracted metrics resolution from core API gateway.

For full historical notes and migration details, use the root changelog.
