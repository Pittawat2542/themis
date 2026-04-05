---
title: Release and versioning
diataxis: project
audience: maintainers
goal: Define how docs quality gates and versioning should work for releases.
---

# Release and versioning

Release expectations:

- no release-ready branch without green docs tests and strict docs build
- public-surface changes must update docs before release
- FAQ and glossary should absorb repeated confusion discovered during release prep or onboarding

Versioning policy:

- before stable release: docs track `main`
- after stable releases begin: publish `latest` plus versioned user docs
- contributor docs continue to track `main`
