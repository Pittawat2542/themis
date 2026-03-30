---
title: Compare, export, and report
diataxis: how-to
audience: users analyzing finished runs
goal: Show how to produce reports, bundle exports, and paired comparisons from stored runs.
---

# Compare, export, and report

Goal: generate portable output from stored runs and compare two completed experiments.

When to use this:

Use this guide when execution is already done and the next task is inspection, reporting, export, or comparison.

## Procedure

Use:

- `Reporter.export_json(...)`, `export_markdown(...)`, `export_csv(...)`, and `export_latex(...)`
- `themis report --config ... --format ...`
- `themis compare --baseline-config ... --candidate-config ...`
- `themis export generation|evaluation --config ...`

## Variants

- one-run reporting: `report`
- portable artifact handoff: `export`
- side-by-side benchmark comparison: `compare`

## Expected result

You should have machine-readable or human-readable output that can be shared without rerunning the experiment.

## Troubleshooting

- [Reporting and read models](../explanation/reporting-and-read-models.md)
- [CLI reference](../reference/cli.md)
