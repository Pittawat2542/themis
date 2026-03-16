# Statistical Comparisons

Use this page when you need to understand what Themis means by a "paired
comparison" before trusting the output of `compare()` or `report()`.

## What Is Paired

Themis compares models on matched rows from the same evaluation overlay. In
practice, the pairing key is:

- `task_id`
- `metric_id`
- `item_id`

If one model is missing a score row for a given item, that item drops out of the
paired comparison. This is why the paired row count can be smaller than the
total number of generated trials.

## What The Reported Numbers Mean

Each comparison row reports:

- baseline mean over the paired items
- treatment mean over the paired items
- `delta_mean`, which is `treatment_mean - baseline_mean`
- bootstrap confidence interval bounds for the paired delta
- raw and adjusted p-values
- `pair_count`, the number of matched items used in the comparison

Treat `pair_count` as a first-pass sanity check. A very small paired set means
the interval and p-value can be unstable even if the point estimate looks large.

## Bootstrap Confidence Intervals

Themis uses paired bootstrap resampling over the matched score rows. The
bootstrap interval answers "how much could the mean delta move if I resampled
the same paired items again?".

Use it to judge effect size uncertainty:

- narrow interval: effect estimate is relatively stable on the observed pairs
- wide interval: the observed delta is sensitive to which items are present
- interval crossing zero: the sign of the effect is not stable on the paired set

## P-Value Corrections

Themis exposes three correction modes:

- `none` for a single comparison or exploratory work
- `holm` when you want family-wise error control across multiple comparisons
- `bh` when you want false-discovery-rate control across many comparisons

Use `holm` by default in shareable reports. Use `bh` when you are screening many
hypotheses and want a less conservative correction.

## When Comparisons Are Invalid or Misleading

Be skeptical of the output when:

- the compared models did not score the same items
- extractors or metrics changed between overlays
- you compare different evaluation overlays unintentionally
- the metric is sparse or heavily missing for one model
- the paired item count is too small for the claim you want to make

For a quick preflight, check:

1. `evaluation_result = result.for_evaluation(result.evaluation_hashes[0])`
2. `comparison = evaluation_result.compare(metric_id="...")`
3. inspect `pair_count`, confidence interval bounds, and adjusted p-value

## Where to Go Next

- Use [Compare and Export Results](../guides/compare-and-export.md) for the
  task-oriented workflow.
- Use [Analyze Results](../guides/analyze-results.md) when you need drilldown
  after finding a suspicious comparison.
- Use [Reporting & Stats API](../api-reference/reporting-and-stats.md) when you
  need exact type and method signatures.
