# Changelog

The canonical release history lives in the repository root:

- [Root changelog on GitHub](https://github.com/Pittawat2542/themis/blob/main/CHANGELOG.md)

## Current Migration Notes

- `parallel_trials` is no longer part of the public orchestration API. Remove
  it from constructor calls and keep only `parallel_candidates`.
- Comparison and report code paths now prefer projection summaries and score
  rows. Custom extensions that only need aggregate data should follow the same
  pattern instead of hydrating every `TrialRecord`.
