# Reporting & Stats

These APIs power `ExperimentResult.report()` and `ExperimentResult.compare()`.

Use them when you need exact exporter types, report-table shapes, or the lower
level statistical engine behind paired comparisons.

The optional `themis.stats` namespace re-exports the paired-comparison engine
behind the `stats` extra. Import from it when you want the short stable path.

::: themis.stats
    options:
      show_root_heading: false

::: themis.report.builder
    options:
      show_root_heading: false

::: themis.report.exporters
    options:
      show_root_heading: false

::: themis.stats.stats_engine
    options:
      show_root_heading: false
